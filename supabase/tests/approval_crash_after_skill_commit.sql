-- Replays the crash window after an immutable Skill commit but before
-- Candidate PROMOTED and Proposal APPROVED. All probe rows roll back.
-- Requires no other pending/processing Proposals (checked by the caller).
DO $approval_crash$
DECLARE
  candidate_id uuid := gen_random_uuid();
  proposal_id text := 'approval-gap-probe-'||candidate_id::text;
  skill_id text := 'approval-gap-skill-'||candidate_id::text;
  old_updated timestamptz;
  recovery jsonb;
  claimed_id text;
  changed integer;
  next_revision bigint;
  n integer;
BEGIN
  BEGIN
    IF EXISTS (
      SELECT 1 FROM public.learning_proposals
      WHERE status IN ('PENDING_APPROVAL','PROCESSING')
    ) THEN
      RAISE EXCEPTION 'An unrelated Proposal is active: abort isolated test';
    END IF;

    INSERT INTO public.learning_candidates
      (id,candidate_key,action,category,status,confidence,occurrences)
    VALUES
      (candidate_id,'approval-gap-probe-'||candidate_id::text,
       'NEW_SKILL','general','ACTIVE',0.9,3);

    INSERT INTO public.learning_proposals
      (id,action,source,status,confidence,analysis,proposed_experience,
       processing_at,created_at)
    VALUES
      (proposal_id,'NEW_SKILL','isolated-db-approval-regression',
       'PROCESSING',0.9,
       jsonb_build_object('candidateMemory',jsonb_build_object('id',candidate_id::text)),
       '{}'::jsonb,now()-interval '61 minutes',now()-interval '61 minutes');

    -- This represents the previously committed Skill. Disable it for
    -- the entire probe; no Skill is published to normal execution.
    INSERT INTO public.jessica_experience_skills
      (id,version,enabled,status,payload)
    VALUES (skill_id,1,false,'draft',
      jsonb_build_object('id',skill_id,'enabled',false));

    INSERT INTO public.jessica_experience_skill_versions
      (skill_id,version,payload)
    VALUES (skill_id,1,jsonb_build_object(
      'id',skill_id,'enabled',false,
      'metadata',jsonb_build_object('proposalId',proposal_id)));

    SELECT updated_at INTO old_updated
      FROM public.learning_candidates WHERE id=candidate_id;
    SELECT count(*)::int INTO n
      FROM public.learning_candidates WHERE id=candidate_id
        AND status='ACTIVE' AND revision=0;
    IF n<>1 THEN RAISE EXCEPTION 'Fixture not in pre-promotion state'; END IF;

    SELECT count(*)::int INTO n
      FROM public.jessica_candidate_skill_publications
      WHERE candidate_id=candidate_id AND proposal_id=proposal_id
        AND skill_id=skill_id AND skill_version=1;
    IF n<>1 THEN RAISE EXCEPTION 'Skill commit and owner claim not atomic'; END IF;

    -- Simulated crash: no Candidate/Proposal finalization above.
    -- Recovery is the real production RPC, not test-local SQL.
    recovery:=public.recover_stale_learning_proposals(30,1);
    IF recovery->>'success' IS DISTINCT FROM 'true'
       OR (recovery->>'recovered')::integer<>1 THEN
      RAISE EXCEPTION 'Stale Proposal did not recover: %',recovery;
    END IF;

    SELECT id INTO claimed_id
      FROM public.claim_learning_proposals(1);
    IF claimed_id IS DISTINCT FROM proposal_id THEN
      RAISE EXCEPTION 'Restart claimed another Proposal: %',claimed_id;
    END IF;

    -- Approval resumes from the saved immutable Proposal receipt.
    SELECT count(*)::int INTO n
      FROM public.jessica_experience_skill_versions
      WHERE skill_id=skill_id AND version=1
        AND payload #>> '{metadata,proposalId}' = proposal_id;
    IF n<>1 THEN RAISE EXCEPTION 'Existing Skill receipt not found'; END IF;

    UPDATE public.learning_candidates
       SET status='PROMOTED',
           metadata=jsonb_build_object('proposalId',proposal_id,'skillId',skill_id),
           updated_at=now()
     WHERE id=candidate_id AND status='ACTIVE'
       AND revision=0 AND updated_at=old_updated
     RETURNING revision INTO next_revision;
    IF next_revision IS DISTINCT FROM 1 THEN
      RAISE EXCEPTION 'Candidate CAS did not promote once: %',next_revision;
    END IF;

    -- Stale Worker cannot repeat Candidate transition.
    UPDATE public.learning_candidates SET status='PROMOTED'
     WHERE id=candidate_id AND status='ACTIVE' AND revision=0;
    GET DIAGNOSTICS changed=ROW_COUNT;
    IF changed<>0 THEN RAISE EXCEPTION 'Candidate promoted twice'; END IF;

    UPDATE public.learning_proposals
       SET status='APPROVED',processing_at=null,approved_at=now()
     WHERE id=proposal_id AND status='PROCESSING';
    GET DIAGNOSTICS changed=ROW_COUNT;
    IF changed<>1 THEN RAISE EXCEPTION 'Proposal approval failed'; END IF;

    UPDATE public.learning_proposals SET status='APPROVED'
     WHERE id=proposal_id AND status='PROCESSING';
    GET DIAGNOSTICS changed=ROW_COUNT;
    IF changed<>0 THEN RAISE EXCEPTION 'Proposal finalized twice'; END IF;

    recovery:=public.recover_stale_learning_proposals(30,1);
    IF (recovery->>'recovered')::integer<>0 THEN
      RAISE EXCEPTION 'Approved Proposal unexpectedly requeued';
    END IF;

    SELECT count(*)::int INTO n
      FROM public.jessica_experience_skill_versions WHERE skill_id=skill_id;
    IF n<>1 THEN RAISE EXCEPTION 'Duplicate Skill version after resume'; END IF;

    SELECT count(*)::int INTO n
      FROM public.learning_candidates c
      JOIN public.learning_proposals p ON p.id=proposal_id
      WHERE c.id=candidate_id AND c.status='PROMOTED' AND c.revision=1
        AND c.metadata->>'proposalId'=proposal_id
        AND c.metadata->>'skillId'=skill_id
        AND p.status='APPROVED';
    IF n<>1 THEN RAISE EXCEPTION 'Broken Candidate-Approval consistency'; END IF;

    -- The test cannot leave Skills or Candidates in production.
    RAISE EXCEPTION 'ROLLBACK_APPROVAL_CRASH_TEST' USING ERRCODE='ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
END
$approval_crash$;
