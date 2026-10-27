-- Renew only a currently owned, non-expired Queue claim.
-- Keep processed_at as the lease clock consumed by recover_stale_learning_queue.
CREATE OR REPLACE FUNCTION public.heartbeat_learning_queue_claim(
  p_queue_item_id text,
  p_claim_token uuid
) RETURNS jsonb
LANGUAGE plpgsql SECURITY INVOKER SET search_path=public
AS $function$
DECLARE v_time timestamptz;
BEGIN
  IF p_claim_token IS NULL OR nullif(btrim(p_queue_item_id),'') IS NULL THEN
    RETURN jsonb_build_object('renewed',false,'reason','invalid');
  END IF;
  UPDATE public.learning_queue q
    SET processed_at=clock_timestamp()
  WHERE q.id=p_queue_item_id
    AND q.status='PROCESSING'
    AND q.claim_token=p_claim_token
    AND q.processed_at IS NOT NULL
    AND q.processed_at >= clock_timestamp()-interval '120 minutes'
  RETURNING q.processed_at INTO v_time;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('renewed',false,'reason','claim_expired_or_stale');
  END IF;
  RETURN jsonb_build_object('renewed',true,'processed_at',v_time);
END $function$;
REVOKE ALL ON FUNCTION public.heartbeat_learning_queue_claim(text,uuid)
  FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.heartbeat_learning_queue_claim(text,uuid)
  TO service_role;