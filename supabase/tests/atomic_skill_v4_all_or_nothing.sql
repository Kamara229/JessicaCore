-- Run directly against Supabase. A single nested subtransaction rolls back
-- every probe artifact after verifying v4 all-or-nothing durability.
-- Intentionally crashes the second module INSERT *after* v3 wrote evidence,
-- current Skill, immutable History, owner claim, and audit.
DO $atomic_skill$
DECLARE
 v_uuid uuid := gen_random_uuid();
 v_candidate uuid := v_uuid;
 v_proposal text := 'atomic-v4-proposal-'||v_uuid::text;
 v_loser text := 'atomic-v4-loser-'||v_uuid::text;
 v_skill text := 'atomic-v4-skill-'||v_uuid::text;
 v_other text := 'atomic-v4-other-'||v_uuid::text;
 v_evidence jsonb;
 v_modules_bad jsonb;
 v_modules_ok jsonb;
 v_payload jsonb;
 v_response jsonb;
 v_blocked boolean := false;
 v_n integer;
BEGIN
 BEGIN
  -- Two Proposals share the same Candidate. All rows exist only in
  -- the outer rollback-only transaction.
  INSERT INTO public.learning_candidates
    (id,candidate_key,action,category,status)
  VALUES (v_candidate,'atomic-v4-candidate-'||v_uuid::text,
      'NEW_SKILL','isolated-regression','ACTIVE');
  INSERT INTO public.learning_proposals(id,action,status,analysis)
  VALUES (v_proposal,'NEW_SKILL','PROCESSING',
    jsonb_build_object('candidateMemory',jsonb_build_object('id',v_candidate::text))),
         (v_loser,'NEW_SKILL','PROCESSING',
    jsonb_build_object('candidateMemory',jsonb_build_object('id',v_candidate::text)));

  v_evidence:=jsonb_build_array(jsonb_build_object(
    'traceId','atomic-v4-evidence-'||v_uuid::text,
    'task','Isolated Skill atomicity regression',
    'success',true,'result',jsonb_build_object('status','COMPLETED')));
  v_modules_bad:=jsonb_build_array(
    jsonb_build_object('moduleKey','workflow','part',0,
      'data',jsonb_build_array('module alpha')),
    jsonb_build_object('moduleKey','workflow','part',0,
      'data',jsonb_build_array('module beta')));
  v_modules_ok:=jsonb_build_array(
    jsonb_build_object('moduleKey','workflow','part',0,
      'data',jsonb_build_array('module alpha')));
  v_payload:=jsonb_build_object(
    'id',v_skill,'name','Isolated disabled atomic Skill',
    'version',1,'enabled',false,
    'moduleManifest',jsonb_build_object(
      'version',1,'counts',jsonb_build_object('workflow',2)),
    'metadata',jsonb_build_object('proposalId',v_proposal));

  -- A duplicate part passes the v4 count check, then fails on the second
  -- jessica_experience_skill_modules PK insert (late post-v3 failure).
  BEGIN
   v_response:=public.save_jessica_experience_skill_v4(
     v_skill,1,NULL,'create',false,v_payload,'{}'::jsonb,
     v_evidence,v_modules_bad);
  EXCEPTION WHEN unique_violation THEN v_blocked:=true;
  END;
  IF NOT v_blocked THEN RAISE EXCEPTION 'Duplicate module part did not fail'; END IF;

  IF EXISTS(SELECT 1 FROM public.jessica_experience_skills WHERE id=v_skill)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_versions WHERE skill_id=v_skill)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_modules WHERE skill_id=v_skill)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_evidence WHERE skill_id=v_skill)
  OR EXISTS(SELECT 1 FROM public.jessica_candidate_skill_publications WHERE candidate_id=v_candidate)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_audit WHERE skill_id=v_skill)
  THEN RAISE EXCEPTION 'Late module insert failure left partial Skill transaction'; END IF;

  -- Retry with valid module count; all dependent tables must commit together.
  v_payload:=jsonb_set(v_payload,'{moduleManifest,counts,workflow}','1'::jsonb);
  v_response:=public.save_jessica_experience_skill_v4(
     v_skill,1,NULL,'create',false,v_payload,'{}'::jsonb,
     v_evidence,v_modules_ok);
  IF v_response->>'success' IS DISTINCT FROM 'true'
     OR v_response->>'existing' IS DISTINCT FROM 'false'
     OR (v_response->>'version')::int <> 1
  THEN RAISE EXCEPTION 'Atomic retry failed: %',v_response; END IF;

  SELECT COUNT(*)::int INTO v_n FROM public.jessica_candidate_skill_publications
    WHERE candidate_id=v_candidate AND proposal_id=v_proposal
    AND skill_id=v_skill AND skill_version=1;
  IF v_n<>1 THEN RAISE EXCEPTION 'Atomic publication claim not committed'; END IF;

  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_versions
    WHERE skill_id=v_skill AND version=1 AND payload #>> '{metadata,proposalId}'=v_proposal;
  IF v_n<>1 THEN RAISE EXCEPTION 'Immutable History not committed'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_modules
    WHERE skill_id=v_skill AND version=1 AND module_key='workflow' AND part_no=0;
  IF v_n<>1 THEN RAISE EXCEPTION 'Valid module not committed'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_evidence
    WHERE skill_id=v_skill AND trace_id='atomic-v4-evidence-'||v_uuid::text;
  IF v_n<>1 THEN RAISE EXCEPTION 'Evidence not committed'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_audit
    WHERE skill_id=v_skill;
  IF v_n<>1 THEN RAISE EXCEPTION 'Audit not committed exactly once'; END IF;

  -- Losing Proposal attempts to publish a different Skill; any early v3
  -- Evidence write must roll back along with its rejected history INSERT.
  v_blocked:=false;
  BEGIN
   PERFORM public.save_jessica_experience_skill_v4(
     v_other,1,NULL,'create',false,
     (v_payload - 'id') || jsonb_build_object(
        'id',v_other,
        'metadata',jsonb_build_object('proposalId',v_loser)),
     '{}'::jsonb,jsonb_build_array(jsonb_build_object(
       'traceId','atomic-v4-loser-evidence-'||v_uuid::text,
       'success',true)),
     v_modules_ok);
  EXCEPTION WHEN unique_violation THEN v_blocked:=true;
  END;
  IF NOT v_blocked THEN RAISE EXCEPTION 'Candidate loser published a Skill'; END IF;
  IF EXISTS(SELECT 1 FROM public.jessica_experience_skills WHERE id=v_other)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_versions WHERE skill_id=v_other)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_modules WHERE skill_id=v_other)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_evidence WHERE skill_id=v_other)
  OR EXISTS(SELECT 1 FROM public.jessica_experience_skill_audit WHERE skill_id=v_other)
  THEN RAISE EXCEPTION 'Rejected competing publication retained partial rows'; END IF;

  -- Idempotent replay must not add a second History/Claim/Module/Audit.
  v_response:=public.save_jessica_experience_skill_v4(
     v_skill,1,NULL,'create',false,v_payload,'{}'::jsonb,
     v_evidence,v_modules_ok);
  IF v_response->>'success' IS DISTINCT FROM 'true'
  OR v_response->>'existing' IS DISTINCT FROM 'true'
  THEN RAISE EXCEPTION 'Saved Skill was republished: %',v_response; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_versions
    WHERE skill_id=v_skill;
  IF v_n<>1 THEN RAISE EXCEPTION 'Duplicate Skill History on replay'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_modules
    WHERE skill_id=v_skill;
  IF v_n<>1 THEN RAISE EXCEPTION 'Duplicate module after replay'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_evidence
    WHERE skill_id=v_skill;
  IF v_n<>1 THEN RAISE EXCEPTION 'Duplicate Evidence after replay'; END IF;
  SELECT COUNT(*)::int INTO v_n FROM public.jessica_experience_skill_audit
    WHERE skill_id=v_skill;
  IF v_n<>1 THEN RAISE EXCEPTION 'Duplicate Audit after replay'; END IF;

  -- Success intentionally rolls back *every* probe table.
  RAISE EXCEPTION 'ROLLBACK_ATOMIC_SKILL_PROBE' USING ERRCODE='ZX001';
 EXCEPTION WHEN SQLSTATE 'ZX001' THEN NULL;
 END;
END
$atomic_skill$;
