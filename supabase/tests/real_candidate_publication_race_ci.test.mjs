import assert from "node:assert/strict";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const winner=sql(`
BEGIN;
SELECT id FROM public.learning_candidates
 WHERE id='8ce0d5d4-f126-472c-942c-9bd18cf52c34' FOR UPDATE;
SELECT pg_sleep(6);
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('ci-candidate-skill-a',1,
 jsonb_build_object('metadata',jsonb_build_object('proposalId','ci-candidate-race-a')));
COMMIT;
`,"jessica-ci-race-winner");

const ownerLocked=await waitFor(`
SELECT CASE WHEN EXISTS(
 SELECT 1 FROM pg_stat_activity WHERE application_name='jessica-ci-race-winner'
 AND state='active' AND query LIKE '%pg_sleep(6)%'
) THEN 1 ELSE 0 END;
`,4000);
assert.equal(ownerLocked,true,"winner must hold Candidate lock before contender starts");

const contender=sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('ci-candidate-skill-b',1,
 jsonb_build_object('metadata',jsonb_build_object('proposalId','ci-candidate-race-b')));
COMMIT;
`,"jessica-ci-race-contender");

const lockObserved=await waitFor(`
SELECT CASE WHEN EXISTS(
 SELECT 1 FROM pg_stat_activity WHERE application_name='jessica-ci-race-contender'
 AND wait_event_type='Lock'
) THEN 1 ELSE 0 END;
`,3500);

const [ownerResult,contenderResult]=await Promise.all([winner,contender]);
assert.equal(lockObserved,true,"contending transaction must wait on a real DB lock");
assert.equal(ownerResult.code,0,ownerResult.stderr);
assert.notEqual(contenderResult.code,0,"second Proposal must not commit");
assert.match(contenderResult.stderr,/Candidate publication: different Skill or Proposal/);

const counts=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
  WHERE proposal_id='ci-candidate-race-a' AND skill_id='ci-candidate-skill-a'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
  WHERE skill_id='ci-candidate-skill-a'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
  WHERE skill_id='ci-candidate-skill-b'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications);
`);
assert.equal(counts.code,0,counts.stderr);
assert.equal(counts.stdout,"1|1|0|1","one owner, one history version, zero loser rows");
console.log("PASS: PostgreSQL row lock + one Candidate publication + zero duplicates");
