-- Resume only a journaled NEW_SKILL when the Candidate row is still ACTIVE.
-- A missing journal or Candidate keeps NEW_SKILL quarantined as FAILED.
-- PATTERN_DISCOVERY remains quarantined: regenerated AI extraction can differ.
-- This overrides the conservative recovery RPC after the durable journal
-- became available; existing proposals still take precedence.
CREATE OR REPLACE FUNCTION public.recover_stale_learning_queue(
  p_stale_minutes integer DEFAULT 120,
  p_limit integer DEFAULT 100
)
RETURNS TABLE(id text, action text, status text)
LANGUAGE sql SECURITY INVOKER SET search_path=public
AS $function$
  WITH locked AS (
    SELECT q.id,q.action
    FROM public.learning_queue q
    WHERE q.status='PROCESSING'
      AND coalesce(q.processed_at,q.created_at)
        < now() - make_interval(mins => greatest(120,least(coalesce(p_stale_minutes,120),10080)))
    ORDER BY coalesce(q.processed_at,q.created_at),q.id
    FOR UPDATE OF q SKIP LOCKED
    LIMIT greatest(1,least(coalesce(p_limit,100),100))
  ),
  classified AS (
    SELECT l.id,CASE
      WHEN EXISTS (
        SELECT 1 FROM public.learning_proposals p WHERE p.queue_item_id=l.id
      ) THEN 'PROPOSED'
      WHEN l.action='SKILL_IMPROVEMENT' THEN 'PENDING'
      WHEN l.action='NEW_SKILL' AND EXISTS (
        SELECT 1
        FROM public.learning_candidate_observation_log j
        JOIN public.learning_candidates c ON c.id=j.candidate_id
        WHERE j.queue_item_id=l.id AND c.status='ACTIVE'
      ) THEN 'PENDING'
      ELSE 'FAILED'
    END AS target_status
    FROM locked l
  ),
  moved AS (
    UPDATE public.learning_queue q
    SET status=c.target_status,claim_token=NULL,
      processed_at=CASE WHEN c.target_status='PENDING' THEN NULL ELSE now() END
    FROM classified c
    WHERE q.id=c.id AND q.status='PROCESSING'
    RETURNING q.id,q.action,q.status
  )
  SELECT moved.id,moved.action,moved.status FROM moved ORDER BY moved.id;
$function$;
REVOKE ALL ON FUNCTION public.recover_stale_learning_queue(integer,integer)
  FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.recover_stale_learning_queue(integer,integer)
  TO service_role;
