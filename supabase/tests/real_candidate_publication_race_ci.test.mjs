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


// Legacy PROMOTED Candidate without a new claim must still be owned by P1.
// PostgreSQL's real guard must deny P2 even when four writes race.
const legacyId="9ff4576b-72bd-4282-a922-4e12ccc1d908";
const legacyP1="ci-legacy-promoted-p1",legacyP2="ci-legacy-promoted-p2";
const legacyS1="ci-legacy-s1",legacyS2="ci-legacy-s2";
const legacySeed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status,metadata)
VALUES('${legacyId}','PROMOTED',
 jsonb_build_object('proposalId','${legacyP1}','skillId','${legacyS1}'));
INSERT INTO public.learning_proposals(id,status,action,analysis)
SELECT p,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${legacyId}'))
FROM (VALUES ('${legacyP1}'),('${legacyP2}')) x(p);
INSERT INTO public.jessica_experience_skills(id)
VALUES('${legacyS1}'),('${legacyS2}');
-- An unlinked historical version predates atomic Candidate claims.
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${legacyS1}',1,jsonb_build_object('id','${legacyS1}',
 'metadata',jsonb_build_object('legacy',true)));
COMMIT;
`);
assert.equal(legacySeed.code,0,legacySeed.stderr);

const legacyClient=(await import("./fixtures/candidateOwnerRealPgClient.mjs"))
  .candidateOwnerRealPgClient;
const {readCandidatePublicationOwner}=await import(
  "../../backend/core/learning/approval/approvalPublicationClaimReader.js");
const {checkLosingCandidatePublication}=await import(
  "../../backend/core/learning/approval/approvalPublicationConflict.js");
const readOwner=id=>readCandidatePublicationOwner(id,{
  client:legacyClient(id)
});
assert.deepEqual(await readOwner(legacyId),{
  proposalId:legacyP1,skillId:legacyS1,evidence:"legacy-promoted-owner"
});
const losing={id:legacyP2,action:"NEW_SKILL",
  candidateMemory:{id:legacyId}};
const conflict=await checkLosingCandidatePublication({
  proposal:losing,readOwner
});
assert.equal(conflict.confirmed,true);
assert.equal(conflict.winningProposalId,legacyP1);
assert.equal(conflict.evidence,"legacy-promoted-owner");

const blockedLegacy=await Promise.all(Array.from({length:4},(_,i)=>sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${legacyS2}',${i+1},
 jsonb_build_object('id','${legacyS2}','metadata',
 jsonb_build_object('proposalId','${legacyP2}')));
COMMIT;
`,"jessica-ci-legacy-loser-"+i)));
for(const r of blockedLegacy){
  assert.notEqual(r.code,0,"P2 must not publish an owned PROMOTED Candidate");
  assert.match(r.stderr,/different Proposal already owns Candidate/);
}
const legacyCounts=await sql(`
SELECT
 (SELECT status FROM public.learning_candidates WHERE id='${legacyId}'),
 (SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${legacyId}'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications WHERE candidate_id='${legacyId}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='${legacyS1}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='${legacyS2}'),
 (SELECT status FROM public.learning_proposals WHERE id='${legacyP2}');
`);
assert.equal(legacyCounts.code,0,legacyCounts.stderr);
assert.equal(legacyCounts.stdout,`PROMOTED|${legacyP1}|0|1|0|PROCESSING`);
console.log("PASS: legacy PROMOTED P1/S1 without claim denies four concurrent P2 Skill writes");
