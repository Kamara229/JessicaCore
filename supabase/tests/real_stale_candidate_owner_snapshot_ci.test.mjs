import assert from "node:assert/strict";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";
import {candidateOwnerRealPgClient} from "./fixtures/candidateOwnerRealPgClient.mjs";
import {readCandidatePublicationOwner}
  from "../../backend/core/learning/approval/approvalPublicationClaimReader.js";
import {checkLosingCandidatePublication}
  from "../../backend/core/learning/approval/approvalPublicationConflict.js";
import {saveApprovedSkillVersion}
  from "../../backend/core/learning/approval/approvalSkillSave.js";

assertIsolatedCi();
const candidateId="85c064e1-6f68-445d-a8df-ae388d3cc95f";
const winner="ci-stale-owner-winner",loser="ci-stale-owner-loser";
const skill="ci-stale-owner-skill";
const seeded=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status) VALUES('${candidateId}','ACTIVE');
INSERT INTO public.learning_proposals(id,status,action,analysis,processing_at)
SELECT id,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidateId}')),now()
FROM (VALUES ('${winner}'),('${loser}')) p(id);
INSERT INTO public.jessica_experience_skills(id) VALUES('${skill}');
COMMIT;
`);
assert.equal(seeded.code,0,seeded.stderr);

// Hold a genuine pre-publication REPEATABLE READ PostgreSQL snapshot while
// another connection commits an atomic Skill claim and Candidate promotion.
const staleSession=sql(`
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidateId}';
SELECT status FROM public.learning_candidates WHERE id='${candidateId}';
SELECT pg_sleep(8);
SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidateId}';
SELECT status FROM public.learning_candidates WHERE id='${candidateId}';
COMMIT;
`,"jessica-ci-stale-owner-snapshot");
const held=await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-stale-owner-snapshot'
 AND state='active' AND query LIKE '%pg_sleep(8)%') THEN 1 ELSE 0 END;
`,3500);
assert.equal(held,true,"read snapshot must begin before publication COMMIT");

const publication=await sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skill}',1,jsonb_build_object('id','${skill}',
 'name','Atomic Skill','version',1,
 'metadata',jsonb_build_object('proposalId','${winner}')));
UPDATE public.learning_candidates
 SET status='PROMOTED',
 metadata=jsonb_build_object('proposalId','${winner}','skillId','${skill}')
 WHERE id='${candidateId}' AND status='ACTIVE';
UPDATE public.learning_proposals
 SET status='APPROVED',approved_at=now(),processing_at=NULL
 WHERE id='${winner}' AND status='PROCESSING';
COMMIT;
`,"jessica-ci-stale-owner-commit");
assert.equal(publication.code,0,publication.stderr);

// Fresh production readback sees the durable new owner immediately.
const readFresh=id=>readCandidatePublicationOwner(id,{
  client:candidateOwnerRealPgClient(id)
});
const fresh=await readFresh(candidateId);
assert.deepEqual(fresh,{
  proposalId:winner,skillId:skill,evidence:"atomic-publication-claim"
});

// A frozen old snapshot still shows the original ACTIVE Candidate with no
// claim. Never treat missing proof as permission to finalize another Proposal.
const stale=await staleSession;
assert.equal(stale.code,0,stale.stderr);
assert.deepEqual(stale.stdout.split("\n").map(x=>x.trim()).filter(Boolean),
  ["0","ACTIVE","0","ACTIVE"],
  "REPEATABLE READ must retain absence despite the committed winner");

const staleRead=id=>{
  assert.equal(id,candidateId);
  return Promise.resolve(null);
};
const losing={id:loser,action:"NEW_SKILL",confidence:0.95,
  candidateMemory:{id:candidateId},
  proposedExperience:{id:"unpublished-skill",name:"Unpublished Skill"}};
const pending=await checkLosingCandidatePublication({
  proposal:losing,readOwner:staleRead
});
assert.equal(pending.confirmed,false);
let failedWrites=0;
const deferred=await saveApprovedSkillVersion({
  proposal:losing,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.95}
},{
  approve:async()=>({success:false,uncertain:true,error:"publication contested"}),
  checkPublicationConflict:({proposal})=>
    checkLosingCandidatePublication({proposal,readOwner:staleRead}),
  markFailed:async()=>{failedWrites++;throw Error("must not finalize stale proof");}
});
assert.equal(deferred.status,"PROCESSING");
assert.equal(deferred.deferred,true);
assert.equal(deferred.proposalStateUpdated,false);
assert.equal(failedWrites,0);

const verified=await checkLosingCandidatePublication({
  proposal:losing,readOwner:readFresh
});
assert.equal(verified.confirmed,true);
assert.equal(verified.winningProposalId,winner);
const final=await sql(`
SELECT json_build_object(
 'claims',(SELECT count(*) FROM public.jessica_candidate_skill_publications
   WHERE candidate_id='${candidateId}'),
 'claim_owner',(SELECT proposal_id FROM public.jessica_candidate_skill_publications
   WHERE candidate_id='${candidateId}'),
 'versions',(SELECT count(*) FROM public.jessica_experience_skill_versions
   WHERE skill_id='${skill}'),
 'candidate_status',(SELECT status FROM public.learning_candidates
   WHERE id='${candidateId}'),
 'loser_status',(SELECT status FROM public.learning_proposals WHERE id='${loser}'),
 'loser_approved',(SELECT approved_at FROM public.learning_proposals WHERE id='${loser}')
)::text;
`);
assert.equal(final.code,0,final.stderr);
const durable=JSON.parse(final.stdout);
assert.deepEqual(durable,{
  claims:1,claim_owner:winner,versions:1,
  candidate_status:"PROMOTED",loser_status:"PROCESSING",loser_approved:null
});
console.log("PASS: old PostgreSQL snapshot cannot falsely finalize losing Proposal");
