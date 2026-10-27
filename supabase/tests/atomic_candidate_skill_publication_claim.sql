-- Safe live-DB regression: the inner subtransaction rolls back all probes.
-- A and B propose different Skills from one ACTIVE Candidate.
DO $test$
DECLARE
  candidate_uuid uuid := gen_random_uuid();
  proposal_a text := 'jessica-claim-a-'||candidate_uuid::text;
  proposal_b text := 'jessica-claim-b-'||candidate_uuid::text;
  skill_a text := 'jessica-claim-skill-a-'||candidate_uuid::text;
  skill_b text := 'jessica-claim-skill-b-'||candidate_uuid::text;
  claims integer;
  blocked boolean := false;
BEGIN
  BEGIN
    INSERT INTO public.learning_candidates
      (id,candidate_key,action,category,status)
    VALUES(candidate_uuid,'claim-probe-'||candidate_uuid::text,
      'NEW_SKILL','general','ACTIVE');

    INSERT INTO public.learning_proposals
      (id,action,status,analysis)
    VALUES
      (proposal_a,'NEW_SKILL','PROCESSING',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',candidate_uuid::text))),
      (proposal_b,'NEW_SKILL','PROCESSING',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',candidate_uuid::text)));

    INSERT INTO public.jessica_experience_skill_versions
      (skill_id,version,payload)
    VALUES(skill_a,1,
      jsonb_build_object('id',skill_a,'metadata',jsonb_build_object('proposalId',proposal_a)));

    BEGIN
      INSERT INTO public.jessica_experience_skill_versions
        (skill_id,version,payload)
      VALUES(skill_b,1,
        jsonb_build_object('id',skill_b,'metadata',jsonb_build_object('proposalId',proposal_b)));
    EXCEPTION WHEN unique_violation THEN
      blocked := true;
    END;

    IF NOT blocked THEN
      RAISE EXCEPTION 'Competing Proposal published a second Skill';
    END IF;

    SELECT count(*)::integer INTO claims
      FROM public.jessica_candidate_skill_publications
      WHERE candidate_id=candidate_uuid AND proposal_id=proposal_a
        AND skill_id=skill_a AND skill_version=1;
    IF claims<>1 THEN
      RAISE EXCEPTION 'Claim not committed with immutable Skill history';
    END IF;

    SELECT count(*)::integer INTO claims
      FROM public.jessica_experience_skill_versions
      WHERE skill_id IN (skill_a,skill_b);
    IF claims<>1 THEN
      RAISE EXCEPTION 'Published more than one Skill';
    END IF;

    -- Terminal state also fences retries from an unrelated Proposal.
    UPDATE public.learning_candidates
      SET status='PROMOTED',
          metadata=jsonb_build_object('proposalId',proposal_a,'skillId',skill_a)
      WHERE id=candidate_uuid;

    blocked:=false;
    BEGIN
      INSERT INTO public.jessica_experience_skill_versions
        (skill_id,version,payload)
      VALUES(skill_b,1,
        jsonb_build_object('id',skill_b,'metadata',jsonb_build_object('proposalId',proposal_b)));
    EXCEPTION WHEN unique_violation THEN
      blocked:=true;
    END;
    IF NOT blocked THEN
      RAISE EXCEPTION 'PROMOTED Candidate was reused';
    END IF;

    -- Force a rollback of all test rows, including the winning publication.
    RAISE EXCEPTION 'ROLLBACK_PROBES' USING ERRCODE='ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
END
$test$;
