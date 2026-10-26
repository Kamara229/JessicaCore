-- Atomic queue ownership: competing Learning Workers never observe one
-- candidate task as claimable at the same time.
CREATE OR REPLACE FUNCTION public.claim_pending_learning_queue(p_limit integer DEFAULT 100)
RETURNS SETOF public.learning_queue
LANGUAGE sql
SECURITY INVOKER
SET search_path = public
AS $function$
  WITH selected AS (
    SELECT id
    FROM public.learning_queue
    WHERE status = 'PENDING'
    ORDER BY created_at, id
    FOR UPDATE SKIP LOCKED
    LIMIT GREATEST(1, LEAST(COALESCE(p_limit, 100), 100))
  ), claimed AS (
    UPDATE public.learning_queue q
    SET status = 'PROCESSING', processed_at = now()
    FROM selected s
    WHERE q.id = s.id AND q.status = 'PENDING'
    RETURNING q.*
  )
  SELECT * FROM claimed ORDER BY created_at, id;
$function$;
REVOKE ALL ON FUNCTION public.claim_pending_learning_queue(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_pending_learning_queue(integer) TO service_role;
