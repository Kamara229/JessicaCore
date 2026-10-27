-- Read-only checkpoint after asynchronous Pattern Discovery.
-- It cannot guarantee a future write; every Candidate and Proposal write is
-- separately fenced under a Queue row lock.
CREATE OR REPLACE FUNCTION public.is_learning_queue_claim_current(
 p_queue_item_id text,p_claim_token uuid
) RETURNS boolean
LANGUAGE sql SECURITY INVOKER SET search_path=public
STABLE AS $function$
 SELECT coalesce(
   p_claim_token IS NOT NULL
   AND EXISTS (
     SELECT 1 FROM public.learning_queue q
      WHERE q.id=p_queue_item_id AND q.status='PROCESSING'
        AND q.claim_token=p_claim_token
   ),false
 );
$function$;
REVOKE ALL ON FUNCTION public.is_learning_queue_claim_current(text,uuid)
 FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.is_learning_queue_claim_current(text,uuid)
 TO service_role;