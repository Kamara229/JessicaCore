import assert from "node:assert/strict";
import {persistExperienceSkill}
 from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {saveApprovedSkillVersion}
 from "../../backend/core/learning/approval/approvalSkillSave.js";
import {checkLosingCandidatePublication}
 from "../../backend/core/learning/approval/approvalPublicationConflict.js";
import {transitionCandidateFinalStatus}
 from "../../backend/core/learning/candidateMemory/candidateFinalStatus.js";
import {transitionProposalStatus}
 from "../../backend/core/learning/proposalStorage/proposalStatusCAS.js";
import {buildProposalStatusUpdate}
 from "../../backend/core/learning/proposalStorage/proposalStorageStatus.js";
import {candidateFinalPgClient} from "./fixtures/candidateFinalPgClient.mjs";
import {proposalStatusClient} from "./fixtures/proposalRestartPgStatusClient.mjs";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const suffix=process.argv[2];
assert.ok(["a","b"].includes(suffix));
const candidateId="d7d26169-ef7b-4d89-ad79-61b7fb22c007";
const proposalId="ci-competing-proposal-"+suffix;
const skillId="ci-competing-skill-"+suffix;
const proposal={
  id:proposalId,action:"NEW_SKILL",confidence:0.97,
  candidateMemory:{id:candidateId},
  proposedExperience:{id:skillId,name:"Independent Skill "+suffix}
};
const lookup=async id=>{
  assert.equal(id,proposalId);
  const found=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${proposalId}' LIMIT 1;
`);
  assert.equal(found.code,0,found.stderr);
  return found.stdout?JSON.parse(found.stdout):null;
};
const save=async value=>{
  const payload=JSON.stringify(value).replaceAll("'","''");
  const result=await sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skillId}',1,'${payload}'::jsonb);
COMMIT;
`,"jessica-ci-competing-insert-"+suffix);
  return result.code===0?
    {success:true,skillId,version:1}:
    {success:false,error:result.stderr};
};
const readOwner=async id=>{
  assert.equal(id,candidateId);
  const r=await sql(`
SELECT json_build_object('proposalId',proposal_id,'skillId',skill_id)::text
FROM public.jessica_candidate_skill_publications
WHERE candidate_id='${candidateId}' LIMIT 1;
`);
  assert.equal(r.code,0,r.stderr);
  return r.stdout?JSON.parse(r.stdout):null;
};
const markFailed=async ({proposal:p,reason})=>{
  assert.equal(p.id,proposalId);
  assert.match(reason,/another Proposal/);
  // The same production terminal CAS must recognize already FAILED status.
  const persistence=await transitionProposalStatus({
    client:proposalStatusClient("restored",p.id),id:p.id,
    status:"FAILED",payload:buildProposalStatusUpdate("FAILED")
  });
  return {persistence,status:"FAILED"};
};
const result=await saveApprovedSkillVersion({
  proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.97}
},{
  approve:async()=>persistExperienceSkill({
    id:skillId,name:"Independent Skill "+suffix,version:1,
    previousVersion:null,metadata:{proposalId}
  },{lookup,save}),
  promote:({proposal:p,skillId:publishedId})=>transitionCandidateFinalStatus({
    client:candidateFinalPgClient(candidateId),id:candidateId,
    status:"PROMOTED",
    metadata:{proposalId:p.id,skillId:publishedId}
  }),
  markApproved:p=>transitionProposalStatus({
    client:proposalStatusClient("restored",p.id),id:p.id,
    status:"APPROVED",payload:buildProposalStatusUpdate("APPROVED")
  }),
  checkPublicationConflict:({proposal:p})=>
    checkLosingCandidatePublication({proposal:p,readOwner}),
  markFailed
});
if(result.success){
  assert.equal(result.status,"APPROVED");
  assert.equal(result.proposalStateUpdated,true);
  console.log("COMPETING_WINNER="+suffix);
}else{
  assert.equal(result.publicationConflict,true,JSON.stringify(result));
  assert.equal(result.deferred,false);
  assert.equal(result.status,"FAILED");
  assert.equal(result.proposalStateUpdated,true);
  assert.equal(result.winningProposalId!==proposalId,true);
  if(result.persistence?.existing===true){
    assert.equal(result.persistence.updated,false);
    console.log("COMPETING_LOSER_REPLAY="+suffix);
  }else{
    assert.equal(result.persistence?.updated,true);
    console.log("COMPETING_LOSER="+suffix);
  }
}
