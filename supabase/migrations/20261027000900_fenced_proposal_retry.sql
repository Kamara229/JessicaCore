CREATE OR REPLACE FUNCTION public.record_learning_proposal_save_failure(p_queue_item_id text, p_max_attempts integer DEFAULT 3)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
DECLARE
  v_item public.learning_queue%ROWTYPE;
  v_proposal text;
  v_attempts integer;
  v_max integer;
  v_state text;
BEGIN
  IF nullif(btrim(p_queue_item_id),'') IS NULL THEN
    RAISE EXCEPTION 'Queue Item ID required' USING ERRCODE='22023';
  END IF;
  v_max := greatest(1,least(coalesce(p_max_attempts,3),5));
  SELECT * INTO v_item FROM public.learning_queue q
    WHERE q.id=p_queue_item_id FOR UPDATE;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('status','not_found','attempts',0,'limit',v_max);
  END IF;
  IF v_item.status <> 'PROCESSING' THEN
    RETURN jsonb_build_object('status','not_processing','attempts',0,
      'queue_status',v_item.status,'limit',v_max);
  END IF;
  IF v_item.claim_token IS NOT NULL THEN
    RETURN jsonb_build_object('status','stale_claim','attempts',0,'limit',v_max);
  END IF;
  SELECT p.id INTO v_proposal
    FROM public.learning_proposals p
    WHERE p.queue_item_id=p_queue_item_id
    ORDER BY p.created_at,p.id LIMIT 1;
  IF v_proposal IS NOT NULL THEN
    UPDATE public.learning_queue
      SET status='PROPOSED',processed_at=now()
      WHERE id=p_queue_item_id;
    RETURN jsonb_build_object('status','already_saved','proposal_id',v_proposal,
      'attempts',0,'limit',v_max);
  END IF;
  -- Without journal proof, a NEW_SKILL retry could double-merge Candidate
  -- evidence. Leave its lease for conservative stale recovery.
  IF v_item.action<>'NEW_SKILL' OR NOT EXISTS (
      SELECT 1
      FROM public.learning_candidate_observation_log j
      JOIN public.learning_candidates c ON c.id=j.candidate_id
      WHERE j.queue_item_id=p_queue_item_id AND c.status='ACTIVE'
    ) THEN
    RETURN jsonb_build_object('status','unverified','attempts',0,'limit',v_max);
  END IF;
  INSERT INTO public.learning_proposal_retry_log(queue_item_id,failure_count)
    VALUES(p_queue_item_id,1)
    ON CONFLICT (queue_item_id) DO UPDATE
    SET failure_count=learning_proposal_retry_log.failure_count+1,
        last_failed_at=now()
    RETURNING failure_count INTO v_attempts;
  v_state:=CASE WHEN v_attempts>=v_max THEN 'FAILED' ELSE 'PENDING' END;
  UPDATE public.learning_queue
    SET status=v_state,
        processed_at=CASE WHEN v_state='PENDING' THEN NULL ELSE now() END
    WHERE id=p_queue_item_id;
  RETURN jsonb_build_object('status',
    CASE WHEN v_state='FAILED' THEN 'exhausted' ELSE 'retry' END,
    'attempts',v_attempts,'limit',v_max,'queue_status',v_state);
END $function$
;
CREATE OR REPLACE FUNCTION public.record_learning_proposal_save_failure_owned(p_queue_item_id text, p_claim_token uuid, p_max_attempts integer DEFAULT 3)
 RETURNS jsonb
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
DECLARE
  v_item public.learning_queue%ROWTYPE;
  v_proposal text;
  v_attempts integer;
  v_max integer;
  v_state text;
BEGIN
  IF nullif(btrim(p_queue_item_id),'') IS NULL THEN
    RAISE EXCEPTION 'Queue Item ID required' USING ERRCODE='22023';
  END IF;
  v_max := greatest(1,least(coalesce(p_max_attempts,3),5));
  SELECT * INTO v_item FROM public.learning_queue q
    WHERE q.id=p_queue_item_id FOR UPDATE;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('status','not_found','attempts',0,'limit',v_max);
  END IF;
  IF v_item.status <> 'PROCESSING' THEN
    RETURN jsonb_build_object('status','not_processing','attempts',0,
      'queue_status',v_item.status,'limit',v_max);
  END IF;
  IF p_claim_token IS NULL OR v_item.claim_token IS DISTINCT FROM p_claim_token THEN
    RETURN jsonb_build_object('status','stale_claim','attempts',0,'limit',v_max);
  END IF;
  SELECT p.id INTO v_proposal
    FROM public.learning_proposals p
    WHERE p.queue_item_id=p_queue_item_id
    ORDER BY p.created_at,p.id LIMIT 1;
  IF v_proposal IS NOT NULL THEN
    UPDATE public.learning_queue
      SET status='PROPOSED',processed_at=now(),claim_token=NULL
      WHERE id=p_queue_item_id;
    RETURN jsonb_build_object('status','already_saved','proposal_id',v_proposal,
      'attempts',0,'limit',v_max);
  END IF;
  -- Without journal proof, a NEW_SKILL retry could double-merge Candidate
  -- evidence. Leave its lease for conservative stale recovery.
  IF v_item.action<>'NEW_SKILL' OR NOT EXISTS (
      SELECT 1
      FROM public.learning_candidate_observation_log j
      JOIN public.learning_candidates c ON c.id=j.candidate_id
      WHERE j.queue_item_id=p_queue_item_id AND c.status='ACTIVE'
    ) THEN
    RETURN jsonb_build_object('status','unverified','attempts',0,'limit',v_max);
  END IF;
  INSERT INTO public.learning_proposal_retry_log(queue_item_id,failure_count)
    VALUES(p_queue_item_id,1)
    ON CONFLICT (queue_item_id) DO UPDATE
    SET failure_count=learning_proposal_retry_log.failure_count+1,
        last_failed_at=now()
    RETURNING failure_count INTO v_attempts;
  v_state:=CASE WHEN v_attempts>=v_max THEN 'FAILED' ELSE 'PENDING' END;
  UPDATE public.learning_queue
    SET status=v_state,claim_token=NULL,
        processed_at=CASE WHEN v_state='PENDING' THEN NULL ELSE now() END
    WHERE id=p_queue_item_id;
  RETURN jsonb_build_object('status',
    CASE WHEN v_state='FAILED' THEN 'exhausted' ELSE 'retry' END,
    'attempts',v_attempts,'limit',v_max,'queue_status',v_state);
END $function$
;
REVOKE ALL ON FUNCTION public.record_learning_proposal_save_failure_owned(text,uuid,integer) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.record_learning_proposal_save_failure_owned(text,uuid,integer) TO service_role;
