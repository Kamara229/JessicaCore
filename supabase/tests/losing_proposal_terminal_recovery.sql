-- Real Postgres regression: the loser cannot publish or return to the queue.
-- An intentional nested-transaction exception rolls every probe row back.
DO $test$
DECLARE
  c uuid := gen_random_uuid();
  winner text := 'terminal-probe-winner-'||c::text;
  loser text := 'terminal-probe-loser-'||c::text;
  a text := 'terminal-probe-skill-a-'||c::text;
  b text := 'terminal-probe-skill-b-'||c::text;
  blocked boolean := false;
  n integer;
  owner text;
BEGIN
  BEGIN
    INSERT INTO public.learning_candidates
      (id,candidate_key,action,category,status)
    VALUES(c,'terminal-probe-'||c::text,'NEW_SKILL','general','ACTIVE');
    INSERT INTO public.learning_proposals
      (id,action,status,analysis,processing_at)
    VALUES
      (winner,'NEW_SKILL','PROCESSING',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',c::text)),now()),
      (loser,'NEW_SKILL','PROCESSING',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',c::text)),now());
    INSERT INTO public.jessica_experience_skills(id,version,payload)
    VALUES(a,1,jsonb_build_object('id',a)),
          (b,1,jsonb_build_object('id',b));
    INSERT INTO public.jessica_experience_skill_versions
      (skill_id,version,payload)
    VALUES(a,1,jsonb_build_object('id',a,
      'metadata',jsonb_build_object('proposalId',winner)));

    BEGIN
      INSERT INTO public.jessica_experience_skill_versions
        (skill_id,version,payload)
      VALUES(b,1,jsonb_build_object('id',b,
        'metadata',jsonb_build_object('proposalId',loser)));
    EXCEPTION WHEN unique_violation THEN blocked:=true;
    END;
    IF NOT blocked THEN
      RAISE EXCEPTION 'Losing Proposal published a second Skill';
    END IF;
    SELECT proposal_id INTO owner
      FROM public.jessica_candidate_skill_publications
      WHERE candidate_id=c;
    IF owner IS DISTINCT FROM winner THEN
      RAISE EXCEPTION 'Claim readback lost the winning Proposal';
    END IF;

    -- Mirrors the backend's atomic PROCESSING->FAILED CAS.
    UPDATE public.learning_proposals
      SET status='FAILED',processing_at=null,
          approved_at=null,rejected_at=null
      WHERE id=loser AND status='PROCESSING';
    GET DIAGNOSTICS n=ROW_COUNT;
    IF n<>1 THEN RAISE EXCEPTION 'Loser was not finalized'; END IF;
    -- Replaying FAILED must not mutate the already-finalized row.
    UPDATE public.learning_proposals
      SET status='FAILED'
      WHERE id=loser AND status='PROCESSING';
    GET DIAGNOSTICS n=ROW_COUNT;
    IF n<>0 THEN RAISE EXCEPTION 'Repeated loser finalization wrote again'; END IF;
    -- Stale recovery/claim transitions operate on PROCESSING/PENDING only.
    UPDATE public.learning_proposals SET status='PENDING_APPROVAL'
      WHERE id=loser AND status='PROCESSING';
    GET DIAGNOSTICS n=ROW_COUNT;
    IF n<>0 THEN RAISE EXCEPTION 'Loser can be recovered into pending'; END IF;
    UPDATE public.learning_proposals SET status='PROCESSING'
      WHERE id=loser AND status='PENDING_APPROVAL';
    GET DIAGNOSTICS n=ROW_COUNT;
    IF n<>0 THEN RAISE EXCEPTION 'Loser can be claimed again'; END IF;

    SELECT count(*)::int INTO n
      FROM public.learning_proposals
      WHERE id=loser AND status='FAILED' AND processing_at IS NULL;
    IF n<>1 THEN RAISE EXCEPTION 'Loser final status not durable'; END IF;
    SELECT count(*)::int INTO n FROM public.learning_proposals
      WHERE id=winner AND status='PROCESSING';
    IF n<>1 THEN RAISE EXCEPTION 'Winner Proposal status changed'; END IF;
    SELECT count(*)::int INTO n FROM public.jessica_experience_skill_versions
      WHERE skill_id IN(a,b);
    IF n<>1 THEN RAISE EXCEPTION 'Unexpected Skill publication count'; END IF;

    RAISE EXCEPTION 'ROLLBACK_TEST_ROWS' USING ERRCODE='ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
END
$test$;
