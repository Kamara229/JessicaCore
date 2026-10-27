-- Status transitions are fenced: only the Worker holding the
-- current PROCESSING claim UUID may commit a queue status change.
CREATE OR REPLACE FUNCTION public.transition_learning_queue_owned(
  p_queue_item_id text,p_claim_token uuid,p_target_status text
) RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER
SET search_path=public AS $f$
DECLARE v_row public.learning_queue%ROWTYPE;
BEGIN
 IF p_claim_token IS NULL OR p_target_status NOT IN
   ('PENDING','PROPOSED','FAILED','IGNORED')
 THEN RETURN jsonb_build_object('updated',false,'reason','invalid'); END IF;
 UPDATE public.learning_queue q
 SET status=p_target_status,claim_token=NULL,
     processed_at=CASE WHEN p_target_status='PENDING' THEN NULL ELSE now() END
 WHERE q.id=p_queue_item_id
   AND q.status='PROCESSING' AND q.claim_token=p_claim_token
 RETURNING * INTO v_row;
 IF NOT FOUND THEN
   RETURN jsonb_build_object('updated',false,'reason','stale_claim');
 END IF;
 RETURN jsonb_build_object('updated',true,'status',v_row.status);
END $f$;
REVOKE ALL ON FUNCTION public.transition_learning_queue_owned(text,uuid,text)
 FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.transition_learning_queue_owned(text,uuid,text)
 TO service_role;