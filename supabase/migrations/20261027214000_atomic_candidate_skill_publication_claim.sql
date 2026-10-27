-- A Candidate may publish at most one Skill from one Proposal.
-- The trigger runs inside the existing v2/v3/v4 atomic Skill save.
-- Claim INSERT and immutable history INSERT either commit together or roll back.
CREATE TABLE IF NOT EXISTS public.jessica_candidate_skill_publications (
    candidate_id uuid PRIMARY KEY REFERENCES public.learning_candidates(id),
    proposal_id text NOT NULL UNIQUE REFERENCES public.learning_proposals(id),
    skill_id text NOT NULL,
    skill_version integer NOT NULL CHECK (skill_version > 0),
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.jessica_candidate_skill_publications ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.jessica_candidate_skill_publications FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.guard_candidate_skill_publication()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = ''
AS $fn$
DECLARE
    v_proposal text;
    v_action text;
    v_candidate text;
    v_row public.learning_candidates%ROWTYPE;
    v_claim public.jessica_candidate_skill_publications%ROWTYPE;
BEGIN
    v_proposal := nullif(btrim(NEW.payload #>> '{metadata,proposalId}'), '');
    IF v_proposal IS NULL THEN RETURN NEW; END IF;

    -- Non-learning Skill saves and legacy unlinked records remain unchanged.
    SELECT action,
      coalesce(nullif(analysis #>> '{candidateMemory,id}', ''),
               nullif(analysis #>> '{provenance,candidateMemoryId}', ''))
      INTO v_action,v_candidate
      FROM public.learning_proposals WHERE id=v_proposal;
    IF NOT FOUND OR v_action IS DISTINCT FROM 'NEW_SKILL'
      OR v_candidate IS NULL THEN RETURN NEW; END IF;

    -- Serializes independent Skill-ID publications for a shared Candidate.
    SELECT * INTO v_row FROM public.learning_candidates
      WHERE id::text=v_candidate FOR UPDATE;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'Candidate publication: referenced Candidate missing'
        USING ERRCODE='23514';
    END IF;
    IF v_row.status NOT IN ('ACTIVE','PROMOTED')
      OR (v_row.status='PROMOTED'
          AND (v_row.metadata->>'proposalId' IS DISTINCT FROM v_proposal
               OR v_row.metadata->>'skillId' IS DISTINCT FROM NEW.skill_id)) THEN
      RAISE EXCEPTION 'Candidate publication: different Proposal already owns Candidate'
        USING ERRCODE='23505';
    END IF;

    INSERT INTO public.jessica_candidate_skill_publications(
      candidate_id,proposal_id,skill_id,skill_version)
    VALUES(v_row.id,v_proposal,NEW.skill_id,NEW.version)
    ON CONFLICT (candidate_id) DO NOTHING;

    SELECT * INTO v_claim
      FROM public.jessica_candidate_skill_publications
      WHERE candidate_id=v_row.id;
    IF NOT FOUND OR v_claim.proposal_id IS DISTINCT FROM v_proposal
      OR v_claim.skill_id IS DISTINCT FROM NEW.skill_id
      OR v_claim.skill_version IS DISTINCT FROM NEW.version THEN
      RAISE EXCEPTION 'Candidate publication: different Skill or Proposal already claimed Candidate'
        USING ERRCODE='23505';
    END IF;
    RETURN NEW;
END
$fn$;

DROP TRIGGER IF EXISTS guard_candidate_skill_publication_on_history
  ON public.jessica_experience_skill_versions;
CREATE TRIGGER guard_candidate_skill_publication_on_history
  AFTER INSERT ON public.jessica_experience_skill_versions
  FOR EACH ROW EXECUTE FUNCTION public.guard_candidate_skill_publication();

REVOKE ALL ON FUNCTION public.guard_candidate_skill_publication() FROM PUBLIC,anon,authenticated;
