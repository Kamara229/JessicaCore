-- Isolated GitHub Actions PostgreSQL fixture; never run against production.
-- Supabase's production migration expects its standard non-login API roles.
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE TABLE public.learning_candidates (
  id uuid PRIMARY KEY,
  status text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE public.learning_proposals (
  id text PRIMARY KEY,
  action text,
  analysis jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE public.jessica_experience_skills (id text PRIMARY KEY);
CREATE TABLE public.jessica_experience_skill_versions (
  skill_id text NOT NULL REFERENCES public.jessica_experience_skills(id),
  version integer NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (skill_id, version)
);
INSERT INTO public.learning_candidates(id,status)
VALUES ('8ce0d5d4-f126-472c-942c-9bd18cf52c34','ACTIVE');
INSERT INTO public.learning_proposals(id,action,analysis)
SELECT proposal,'NEW_SKILL',jsonb_build_object('candidateMemory',
 jsonb_build_object('id','8ce0d5d4-f126-472c-942c-9bd18cf52c34'))
FROM (VALUES ('ci-candidate-race-a'),('ci-candidate-race-b')) AS p(proposal);
INSERT INTO public.jessica_experience_skills(id)
VALUES ('ci-candidate-skill-a'),('ci-candidate-skill-b');
