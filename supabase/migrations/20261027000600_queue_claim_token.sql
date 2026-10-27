ALTER TABLE public.learning_queue
  ADD COLUMN IF NOT EXISTS claim_token uuid;

CREATE OR REPLACE FUNCTION public.claim_pending_learning_queue(p_limit integer DEFAULT 100)
RETURNS SETOF public.learning_queue
LANGUAGE sql SECURITY INVOKER SET search_path=public
AS $f$
 WITH selected AS (
   SELECT q.id FROM public.learning_queue q
   WHERE q.status='PENDING'
   ORDER BY q.created_at,q.id
   FOR UPDATE OF q SKIP LOCKED
   LIMIT greatest(1,least(coalesce(p_limit,100),100))
 ), claimed AS (
   UPDATE public.learning_queue q
   SET status='PROCESSING',processed_at=now(),claim_token=gen_random_uuid()
   FROM selected s
   WHERE q.id=s.id AND q.status='PENDING'
   RETURNING q.*
 )
 SELECT * FROM claimed ORDER BY created_at,id;
$f$;
REVOKE ALL ON FUNCTION public.claim_pending_learning_queue(integer) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.claim_pending_learning_queue(integer) TO service_role;
