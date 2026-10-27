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
import {legacyClaimReceiptLookup} from "./fixtures/legacyClaimReceiptPg.mjs";

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
let attempts=0;
const receipt=legacyClaimReceiptLookup(owner);
const lookup=id=>receipt.lookup(id);
const save=async value=>{
 attempts++;
 await raceLegacyClaimDuringCommit({
  candidate,owner,skill,loser,loserSkill,payload:value
 });
 // The commit is durable; both the ACK and receipt DB connection disappear.
 receipt.setOutage(true);
 // The real transaction committed, but transport drops its ACK.
 throw Error("legacy migration commit ACK lost");
};

const lost=await persistExperienceSkill(experience,{lookup,save});
assert.equal(lost.success,false);
assert.equal(lost.uncertain,true);
assert.equal(lost.existing,false);
assert.equal(attempts,1,"the write committed before the ACK was lost");
assert.ok(receipt.calls>=2);

// Repeated workers must not retry writes when the PostgreSQL readback
// connection is unavailable, regardless of the committed claim.
const pending=await Promise.all(Array.from({length:8},()=>
 persistExperienceSkill(experience,{lookup,save})));
assert.ok(pending.every(r=>r.success===false&&r.uncertain===true));
assert.equal(attempts,1,"offline recovery cannot generate an extra Skill version");
assert.deepEqual(await readOwner(candidate),{
 proposalId:owner,skillId:skill,evidence:"atomic-publication-claim"
});

// A new recovery worker with restored PostgreSQL readback accepts only the
// immutable, identity-validated receipt for the first committed version.
receipt.setOutage(false);
const recovered=await persistExperienceSkill(experience,{lookup,save});
assert.equal(recovered.success,true);
assert.equal(recovered.existing,true);
assert.equal(recovered.skillId,skill);
assert.equal(recovered.version,2);
assert.equal(recovered.storage,"idempotency-precheck");
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
console.log("PASS: competing legacy COMMIT lost ACK, 9 failed-closed readbacks, 16 safe replays");
