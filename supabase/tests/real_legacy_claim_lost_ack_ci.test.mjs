import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {readCandidatePublicationOwner}
  from "../../backend/core/learning/approval/approvalPublicationClaimReader.js";
import {checkLosingCandidatePublication}
  from "../../backend/core/learning/approval/approvalPublicationConflict.js";
import {candidateOwnerRealPgClient} from "./fixtures/candidateOwnerRealPgClient.mjs";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";
import {raceLegacyClaimDuringCommit} from "./fixtures/legacyClaimAckRacePg.mjs";

assertIsolatedCi();
const candidate="ea103772-d67b-4a0d-a6e7-8cc0bb3f849b";
const owner="ci-legacy-ack-owner",loser="ci-legacy-ack-loser";
const skill="ci-legacy-ack-skill",loserSkill="ci-legacy-ack-loser-skill";
const seeded=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status,metadata)
VALUES('${candidate}','PROMOTED',
 jsonb_build_object('proposalId','${owner}','skillId','${skill}'));
INSERT INTO public.learning_proposals(id,status,action,analysis)
SELECT id,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}'))
FROM (VALUES ('${owner}'),('${loser}')) p(id);
INSERT INTO public.jessica_experience_skills(id)
VALUES('${skill}'),('${loserSkill}');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skill}',1,jsonb_build_object('id','${skill}',
 'metadata',jsonb_build_object('legacy',true)));
COMMIT;
`);
assert.equal(seeded.code,0,seeded.stderr);

const readOwner=id=>readCandidatePublicationOwner(id,{
 client:candidateOwnerRealPgClient(id)
});
assert.deepEqual(await readOwner(candidate),{
 proposalId:owner,skillId:skill,evidence:"legacy-promoted-owner"
});
const experience={
 id:skill,name:"Legacy owner claim migration with lost ACK",
 version:2,previousVersion:1,mode:"update",enabled:false,
 workflow:["Confirm unique durable Skill claim on restart"],
 metadata:{proposalId:owner,candidateMemoryId:candidate}
};
let attempts=0,lookups=0;
const lookup=async id=>{
 assert.equal(id,owner);
 lookups++;
 const r=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${owner}'
ORDER BY version DESC LIMIT 1;
`);
 assert.equal(r.code,0,r.stderr);
 return r.stdout?JSON.parse(r.stdout):null;
};
const save=async value=>{
 attempts++;
 await raceLegacyClaimDuringCommit({
  candidate,owner,skill,loser,loserSkill,payload:value
 });
 // The real transaction committed, but transport drops its ACK.
 throw Error("legacy migration commit ACK lost");
};

const saved=await persistExperienceSkill(experience,{lookup,save});
assert.equal(saved.success,true);
assert.equal(saved.existing,true);
assert.equal(saved.commitRecovered,true);
assert.equal(saved.skillId,skill);
assert.equal(saved.version,2);
assert.equal(attempts,1);
assert.ok(lookups>=2);
assert.deepEqual(await readOwner(candidate),{
 proposalId:owner,skillId:skill,evidence:"atomic-publication-claim"
});

// Independent retries after losing the COMMIT acknowledgement must recover
// the same receipt; none may publish another immutable version.
const replayed=await Promise.all(Array.from({length:16},()=>
 persistExperienceSkill(experience,{lookup,save})));
assert.equal(replayed.length,16);
assert.ok(replayed.every(r=>r.success&&r.existing&&
 r.skillId===skill&&r.version===2));
assert.equal(attempts,1);

const denied=await sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${loserSkill}',1,jsonb_build_object('id','${loserSkill}',
 'metadata',jsonb_build_object('proposalId','${loser}')));
COMMIT;
`,"jessica-ci-legacy-claim-loser");
assert.notEqual(denied.code,0);
assert.match(denied.stderr,/different Proposal already owns Candidate/);
const conflict=await checkLosingCandidatePublication({
 proposal:{id:loser,action:"NEW_SKILL",candidateMemory:{id:candidate}},
 readOwner
});
assert.equal(conflict.confirmed,true);
assert.equal(conflict.winningProposalId,owner);

const final=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 (SELECT proposal_id FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 (SELECT skill_version FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='${skill}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='${loserSkill}'),
 (SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${candidate}');
`);
assert.equal(final.code,0,final.stderr);
assert.equal(final.stdout,`1|${owner}|2|2|0|${owner}`);
console.log("PASS: lost legacy migration ACK recovers one claim; 16 replays no duplicates");
