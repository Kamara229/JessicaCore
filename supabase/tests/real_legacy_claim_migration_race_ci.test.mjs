import assert from "node:assert/strict";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";
import {candidateOwnerRealPgClient} from "./fixtures/candidateOwnerRealPgClient.mjs";
import {readCandidatePublicationOwner}
  from "../../backend/core/learning/approval/approvalPublicationClaimReader.js";

assertIsolatedCi();
const candidate="d3f28670-ac50-4ba5-afb4-0581c24e1a04";
const p1="ci-migration-owner-p1",p2="ci-migration-intruder-p2";
const s1="ci-migration-skill-s1",s2="ci-migration-skill-s2";
const seeded=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status,metadata)
VALUES('${candidate}','PROMOTED',
 jsonb_build_object('proposalId','${p1}','skillId','${s1}'));
INSERT INTO public.learning_proposals(id,status,action,analysis)
SELECT p,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}'))
FROM (VALUES ('${p1}'),('${p2}')) a(p);
INSERT INTO public.jessica_experience_skills(id) VALUES('${s1}'),('${s2}');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${s1}',1,jsonb_build_object('metadata',
 jsonb_build_object('legacy',true)));
COMMIT;
`);
assert.equal(seeded.code,0,seeded.stderr);
const ownerRead=()=>readCandidatePublicationOwner(candidate,{
  client:candidateOwnerRealPgClient(candidate)
});
assert.deepEqual(await ownerRead(),{
  proposalId:p1,skillId:s1,evidence:"legacy-promoted-owner"
});

const initial=await sql(`
SELECT count(*) FROM public.jessica_candidate_skill_publications
WHERE candidate_id='${candidate}';
`);
assert.equal(initial.code,0,initial.stderr);
assert.equal(initial.stdout,"0");

// Three independent PostgreSQL sessions compete for a real Candidate row
// lock, with no claim before P1's migration transaction commits.
const winner=sql(`
BEGIN;
SELECT id FROM public.learning_candidates WHERE id='${candidate}' FOR UPDATE;
SELECT pg_sleep(6);
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${s1}',2,jsonb_build_object('metadata',
 jsonb_build_object('proposalId','${p1}')));
COMMIT;
`,"jessica-ci-legacy-migration-winner");
assert.equal(await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-legacy-migration-winner'
 AND state='active' AND query LIKE '%pg_sleep(6)%') THEN 1 ELSE 0 END;
`,4000),true,"P1 must hold the Candidate lock before racing writers");

function contender(app,skill,version,proposal){
  return sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skill}',${version},jsonb_build_object('metadata',
 jsonb_build_object('proposalId','${proposal}')));
COMMIT;
`,app);
}
const repeat=contender("jessica-ci-legacy-migration-repeat",s1,3,p1);
const theft=contender("jessica-ci-legacy-migration-theft",s2,1,p2);
const blocked=await waitFor(`
SELECT CASE WHEN count(*)=2 THEN 1 ELSE 0 END
FROM pg_stat_activity
WHERE application_name IN ('jessica-ci-legacy-migration-repeat',
 'jessica-ci-legacy-migration-theft') AND wait_event_type='Lock';
`,5000);
const [won,repeated,stolen]=await Promise.all([winner,repeat,theft]);
assert.equal(blocked,true,"both writers must contend on actual PostgreSQL locks");
assert.equal(won.code,0,won.stderr);
assert.notEqual(repeated.code,0,"P1 different version must not overwrite immutable claim");
assert.match(repeated.stderr,/different Skill or Proposal already claimed Candidate/);
assert.notEqual(stolen.code,0,"P2 must not steal PROMOTED Candidate");
assert.match(stolen.stderr,/different Proposal already owns Candidate/);

assert.deepEqual(await ownerRead(),{
  proposalId:p1,skillId:s1,evidence:"atomic-publication-claim"
});
const state=async()=>{
  const r=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT proposal_id FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT skill_version FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${s1}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${s2}'),
 (SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 (SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${candidate}');
`);
  assert.equal(r.code,0,r.stderr);
  return r.stdout;
};
const snapshot=await state();
assert.equal(snapshot,`1|${p1}|2|2|0|PROMOTED|${p1}`);
const retry=await contender("jessica-ci-legacy-migration-retry",s1,2,p1);
assert.notEqual(retry.code,0,"same version retry must be duplicate-safe");
assert.deepEqual(await state(),snapshot);
console.log("PASS: legacy P1 migration serializes two contenders, single claim, no stolen owner");
