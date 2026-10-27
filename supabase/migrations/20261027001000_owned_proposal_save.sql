-- A delayed Worker must not insert a Proposal after its lease expired.
-- The Queue row lock serializes insert with stale recovery and status writes.
CREATE OR REPLACE FUNCTION public.save_learning_proposal_owned(
  p_queue_item_id text,p_claim_token uuid,p_payload jsonb
) RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER
SET search_path=public AS $f$
DECLARE v_q public.learning_queue%ROWTYPE;
        v_p public.learning_proposals%ROWTYPE;
BEGIN
 IF p_claim_token IS NULL OR jsonb_typeof(p_payload)<>'object'
    OR nullif(p_payload->>'id','') IS NULL
    OR p_payload->>'queue_item_id' IS DISTINCT FROM p_queue_item_id
    OR p_payload->>'status' IS DISTINCT FROM 'PENDING_APPROVAL'
    OR jsonb_typeof(p_payload->'proposed_experience')<>'object'
 THEN RETURN jsonb_build_object('saved',false,'reason','invalid'); END IF;
 SELECT * INTO v_q FROM public.learning_queue
  WHERE id=p_queue_item_id FOR UPDATE;
 IF NOT FOUND OR v_q.status<>'PROCESSING'
    OR v_q.claim_token IS DISTINCT FROM p_claim_token
 THEN RETURN jsonb_build_object('saved',false,'reason','stale_claim'); END IF;
 IF p_payload->>'action' NOT IN ('NEW_SKILL','SKILL_IMPROVEMENT')
   OR (v_q.action <> p_payload->>'action'
        AND NOT (v_q.action='PATTERN_DISCOVERY'
                 AND p_payload->>'action'='NEW_SKILL'))
 THEN RETURN jsonb_build_object('saved',false,'reason','action_mismatch'); END IF;
 INSERT INTO public.learning_proposals
   (id,queue_item_id,status,source,action,confidence,
    proposed_experience,analysis,target_skill,created_at)
 VALUES
   (p_payload->>'id',p_queue_item_id,'PENDING_APPROVAL',
    coalesce(nullif(p_payload->>'source',''),'learning_queue'),
    p_payload->>'action',coalesce((p_payload->>'confidence')::double precision,0),
    coalesce(p_payload->'proposed_experience','{}'::jsonb),
    coalesce(p_payload->'analysis','{}'::jsonb),
    coalesce(p_payload->'target_skill','{}'::jsonb),
    coalesce((p_payload->>'created_at')::timestamptz,now()))
 ON CONFLICT (queue_item_id) WHERE queue_item_id IS NOT NULL DO NOTHING
 RETURNING * INTO v_p;
 IF FOUND THEN
  RETURN jsonb_build_object('saved',true,'existing',false,'proposal',to_jsonb(v_p));
 END IF;
 SELECT * INTO v_p FROM public.learning_proposals
  WHERE queue_item_id=p_queue_item_id;
 IF NOT FOUND THEN
  RETURN jsonb_build_object('saved',false,'reason','proposal_missing');
 END IF;
 RETURN jsonb_build_object('saved',true,'existing',true,'proposal',to_jsonb(v_p));
END $f$;
REVOKE ALL ON FUNCTION public.save_learning_proposal_owned(text,uuid,jsonb)
  FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.save_learning_proposal_owned(text,uuid,jsonb)
  TO service_role;