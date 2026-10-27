-- Crash/restart simulation against the deployed PostgreSQL RPCs.
-- A nested subtransaction rolls back every fixture on completion.
DO $test$
DECLARE
  c uuid := gen_random_uuid();
  winner text := 'restart-probe-winner-'||c::text;
  loser text := 'restart-probe-loser-'||c::text;
  skill text := 'restart-probe-skill-'||c::text;
  outcome jsonb;
  claimed record;
  n integer;
BEGIN
  BEGIN
    -- Simulate the pre-restart database: Skill and Candidate committed,
    -- Proposal status ACK was lost, and a different loser is terminal.
    INSERT INTO public.learning_candidates
      (id,candidate_key,action,category,status)
    VALUES(c,'restart-probe-'||c::text,'NEW_SKILL','general','ACTIVE');
    INSERT INTO public.learning_proposals
      (id,action,status,analysis,processing_at)
    VALUES
      (winner,'NEW_SKILL','PROCESSING',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',c::text)),
       now()-interval '61 minutes'),
      (loser,'NEW_SKILL','FAILED',
       jsonb_build_object('candidateMemory',jsonb_build_object('id',c::text)),
       null);
    INSERT INTO public.jessica_experience_skills(id,version,payload)
    VALUES(skill,1,jsonb_build_object('id',skill));
    INSERT INTO public.jessica_experience_skill_versions
      (skill_id,version,payload)
    VALUES(skill,1,jsonb_build_object('id',skill,
       'metadata',jsonb_build_object('proposalId',winner)));
    UPDATE public.learning_candidates SET status='PROMOTED',
      metadata=jsonb_build_object('proposalId',winner,'skillId',skill)
      WHERE id=c;

    -- The restarted daemon's first cycle runs stale recovery.
    outcome := public.recover_stale_learning_proposals(30,1);
    IF outcome->>'success' IS DISTINCT FROM 'true' OR
       (outcome->>'recovered')::integer<>1 THEN
       RAISE EXCEPTION 'Restart did not recover the sole stale Proposal: %',outcome;
    END IF;

    SELECT count(*)::int INTO n FROM public.learning_proposals
      WHERE id=winner AND status='PENDING_APPROVAL';
    IF n<>1 THEN RAISE EXCEPTION 'Winner was not made claimable'; END IF;

    -- Following atomic claim must return the same persisted Proposal.
    n:=0;
    FOR claimed IN SELECT * FROM public.claim_learning_proposals(1) LOOP
      IF claimed.id IS DISTINCT FROM winner OR
         claimed.status IS DISTINCT FROM 'PROCESSING' THEN
        RAISE EXCEPTION 'Restart claimed an unrelated Proposal';
      END IF;
      n:=n+1;
    END LOOP;
    IF n<>1 THEN RAISE EXCEPTION 'Restart claim did not return exactly one Proposal'; END IF;

    -- Repeated Approval can read immutable Skill history and Candidate
    -- owner instead of attempting another Skill publication.
    SELECT count(*)::int INTO n
      FROM public.jessica_experience_skill_versions
      WHERE skill_id=skill AND payload #>> '{metadata,proposalId}'=winner;
    IF n<>1 THEN RAISE EXCEPTION 'Immutable Skill receipt is missing or duplicated'; END IF;
    SELECT count(*)::int INTO n
      FROM public.jessica_candidate_skill_publications
      WHERE candidate_id=c AND proposal_id=winner AND skill_id=skill;
    IF n<>1 THEN RAISE EXCEPTION 'Candidate owner changed after restart'; END IF;

    -- The Approval status CAS finishes; terminal loser stays untouched.
    UPDATE public.learning_proposals
      SET status='APPROVED',processing_at=null,approved_at=now()
      WHERE id=winner AND status='PROCESSING';
    GET DIAGNOSTICS n=ROW_COUNT;
    IF n<>1 THEN RAISE EXCEPTION 'Approved Proposal did not finalize'; END IF;

    outcome:=public.recover_stale_learning_proposals(30,1);
    IF (outcome->>'recovered')::integer<>0 THEN
      RAISE EXCEPTION 'Restart recovery replayed a terminal Proposal';
    END IF;
    SELECT count(*)::int INTO n FROM public.learning_proposals
      WHERE id=loser AND status='FAILED' AND processing_at IS NULL;
    IF n<>1 THEN RAISE EXCEPTION 'Losing Proposal was requeued'; END IF;
    SELECT count(*)::int INTO n FROM public.jessica_experience_skill_versions
      WHERE skill_id=skill;
    IF n<>1 THEN RAISE EXCEPTION 'Restart republished a Skill version'; END IF;

    RAISE EXCEPTION 'ROLLBACK_RESTART_PROBES' USING ERRCODE='ZX001';
  EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
  END;
END
$test$;