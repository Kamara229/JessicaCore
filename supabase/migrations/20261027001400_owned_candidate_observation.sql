-- Owner token and queue row lock serialize Candidate updates with lease recovery.
CREATE OR REPLACE FUNCTION public.apply_candidate_memory_observation_owned(p_queue_item_id text, p_candidate_id uuid, p_candidate_key text, p_action text, p_skill_id text, p_category text, p_candidate jsonb, p_metadata jsonb, p_expected_occurrences integer, p_claim_token uuid)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
DECLARE
  v_inserted text;
  v_prior_candidate uuid;
  v_row public.learning_candidates%ROWTYPE;
  v_occ integer;
  v_owner public.learning_queue%ROWTYPE;
BEGIN
  IF nullif(btrim(p_queue_item_id),'') IS NULL OR p_candidate_id IS NULL
    OR nullif(btrim(p_candidate_key),'') IS NULL
    OR p_action NOT IN ('NEW','EXACT','SIMILARITY')
    OR jsonb_typeof(p_candidate) <> 'object'
    OR jsonb_typeof(p_metadata) <> 'object'
  THEN RAISE EXCEPTION 'Invalid candidate observation' USING ERRCODE='22023'; END IF;
  IF p_claim_token IS NULL THEN
    RAISE EXCEPTION 'Current Queue claim token is required' USING ERRCODE='55000';
  END IF;
  SELECT * INTO v_owner
  FROM public.learning_queue q
  WHERE q.id=p_queue_item_id
  FOR UPDATE;
  IF NOT FOUND OR v_owner.status <> 'PROCESSING'
    OR v_owner.claim_token IS DISTINCT FROM p_claim_token THEN
    RAISE EXCEPTION 'Stale Candidate Memory Queue ownership' USING ERRCODE='55000';
  END IF;
  v_occ := COALESCE(NULLIF(p_candidate->>'occurrences','')::integer,1);
  IF v_occ < 1 OR v_occ > 1000000 THEN
    RAISE EXCEPTION 'Invalid candidate occurrence count' USING ERRCODE='22023';
  END IF;
  -- PostgreSQL serializes ON CONFLICT against concurrent requests of same
  -- queue_item_id. Any downstream failure rolls this log record back too.
  INSERT INTO public.learning_candidate_observation_log(queue_item_id,candidate_id,action)
  VALUES(p_queue_item_id,p_candidate_id,p_action)
  ON CONFLICT (queue_item_id) DO NOTHING
  RETURNING queue_item_id INTO v_inserted;
  IF v_inserted IS NULL THEN
    SELECT candidate_id INTO v_prior_candidate
    FROM public.learning_candidate_observation_log
    WHERE queue_item_id=p_queue_item_id;
    IF v_prior_candidate IS NULL THEN
      RAISE EXCEPTION 'Candidate observation replay lookup unavailable';
    END IF;
    SELECT * INTO v_row FROM public.learning_candidates
      WHERE id=v_prior_candidate;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'Candidate observation journal points to missing candidate';
    END IF;
    RETURN jsonb_build_object('applied',false,'existing',true,'candidate',to_jsonb(v_row));
  END IF;
  IF p_action='NEW' THEN
    IF p_expected_occurrences IS NOT NULL THEN
      RAISE EXCEPTION 'Unexpected expected occurrences for NEW';
    END IF;
    INSERT INTO public.learning_candidates
      (id,candidate_key,action,skill_id,category,status,confidence,occurrences,
       candidate_json,metadata,created_at,updated_at,last_seen_at)
    VALUES
      (p_candidate_id,p_candidate_key,'NEW_SKILL',nullif(btrim(p_skill_id),''),
       coalesce(nullif(btrim(p_category),''),'general'),'ACTIVE',
       coalesce((p_candidate->>'confidence')::double precision,0),
       v_occ,p_candidate,p_metadata,now(),now(),now())
    RETURNING * INTO v_row;
  ELSE
    IF p_expected_occurrences IS NULL THEN
      RAISE EXCEPTION 'Missing expected candidate occurrences';
    END IF;
    SELECT * INTO v_row FROM public.learning_candidates
      WHERE id=p_candidate_id FOR UPDATE;
    IF NOT FOUND OR v_row.status <> 'ACTIVE'
       OR v_row.candidate_key <> p_candidate_key THEN
      RAISE EXCEPTION 'Candidate observation target changed';
    END IF;
    IF v_row.occurrences <> p_expected_occurrences THEN
      RAISE EXCEPTION 'Stale candidate observation: expected %, got %',
        p_expected_occurrences,v_row.occurrences
      USING ERRCODE='40001';
    END IF;
    IF v_occ < v_row.occurrences THEN
      RAISE EXCEPTION 'Candidate occurrence count cannot decrease';
    END IF;
    UPDATE public.learning_candidates SET
      candidate_json=p_candidate,
      confidence=coalesce((p_candidate->>'confidence')::double precision,0),
      occurrences=v_occ,
      skill_id=coalesce(nullif(btrim(p_skill_id),''),skill_id),
      metadata=p_metadata,updated_at=now(),last_seen_at=now()
    WHERE id=p_candidate_id
    RETURNING * INTO v_row;
  END IF;
  RETURN jsonb_build_object('applied',true,'existing',false,'candidate',to_jsonb(v_row));
END $function$
;
-- The legacy nine-argument RPC may not bypass Queue ownership after deployment.
CREATE OR REPLACE FUNCTION public.apply_candidate_memory_observation(
 p_queue_item_id text, p_candidate_id uuid, p_candidate_key text,
 p_action text,p_skill_id text,p_category text,p_candidate jsonb,
 p_metadata jsonb,p_expected_occurrences integer DEFAULT NULL
) RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=public AS $legacy$
BEGIN
 RAISE EXCEPTION 'Queue claim token required; use owned Candidate observation RPC'
 USING ERRCODE='55000';
END $legacy$;
REVOKE ALL ON FUNCTION public.apply_candidate_memory_observation_owned(text,uuid,text,text,text,text,jsonb,jsonb,integer,uuid)
 FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.apply_candidate_memory_observation_owned(text,uuid,text,text,text,text,jsonb,jsonb,integer,uuid)
 TO service_role;