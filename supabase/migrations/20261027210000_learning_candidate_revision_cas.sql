-- Monotonic Candidate revision fences racing consolidation and finalization.
-- Covers every writer, including existing owned observation RPCs.
ALTER TABLE public.learning_candidates
  ADD COLUMN IF NOT EXISTS revision bigint NOT NULL DEFAULT 0;

CREATE OR REPLACE FUNCTION public.bump_learning_candidate_revision()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER SET search_path=public
AS $function$
BEGIN
  NEW.revision := OLD.revision + 1;
  RETURN NEW;
END
$function$;

CREATE TRIGGER bump_learning_candidate_revision_on_update
BEFORE UPDATE ON public.learning_candidates
FOR EACH ROW EXECUTE FUNCTION public.bump_learning_candidate_revision();

REVOKE ALL ON FUNCTION public.bump_learning_candidate_revision()
  FROM PUBLIC, anon, authenticated;
