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
const fault=process.argv[3]||"normal";
const scenario=process.env.JESSICA_COMPETING_ISOLATED_SCENARIO||"";
const outageScenario=scenario==="readback-outage";
const ownerOutageScenario=scenario==="owner-readback-outage";
assert.ok(["a","b"].includes(suffix));
assert.ok(["normal","lost-failed-ack","lost-failed-ack-offline",
  "readback-offline","owner-offline"].includes(fault));
assert.ok(!scenario||outageScenario||ownerOutageScenario);
assert.ok(fault!=="owner-offline"||ownerOutageScenario);
const loseFailedAck=fault.startsWith("lost-failed-ack");
const candidateId=ownerOutageScenario
  ?"bea2b33c-1a6a-42fd-9693-61cbd2c865b0"
  :outageScenario?"aa4f2c21-c621-45db-b71e-33f9a5e8bd10"
  :"d7d26169-ef7b-4d89-ad79-61b7fb22c007";
const prefix=ownerOutageScenario?"ci-owner-outage-":
  outageScenario?"ci-failed-outage-":"ci-competing-";
const proposalId=prefix+"proposal-"+suffix;
const skillId=prefix+"skill-"+suffix;
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
`,"jessica-ci-owner-readback",
    fault==="owner-offline"?{PGPORT:"1"}:{});
  if(r.code!==0)throw Error("Candidate owner readback unavailable: "+r.stderr);
  return r.stdout?JSON.parse(r.stdout):null;
};
const markFailed=async ({proposal:p,reason})=>{
  assert.equal(p.id,proposalId);
  assert.match(reason,/another Proposal/);
  // The same production terminal CAS must recognize already FAILED status.
  const persistence=await transitionProposalStatus({
    client:proposalStatusClient(
      fault==="lost-failed-ack-offline"?"lost-ack":
      fault==="readback-offline"?"offline":
      loseFailedAck?"lost-ack-readable":"restored",p.id),
    id:p.id,status:"FAILED",payload:buildProposalStatusUpdate("FAILED")
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
}else if(fault==="owner-offline"){
  assert.equal(result.success,false);
  assert.equal(result.deferred,true,"unknown owner must not finalize Proposal");
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.publicationConflict===true,false,"no unverified conflict");
  assert.equal(result.proposalStateUpdated,false);
  assert.equal(result.candidateMemoryStateUpdated,false);
  console.log("COMPETING_OWNER_UNKNOWN="+suffix);
}else{
  assert.equal(result.publicationConflict,true,JSON.stringify(result));
  assert.equal(result.winningProposalId!==proposalId,true);
  if(["lost-failed-ack-offline","readback-offline"].includes(fault)){
    assert.equal(result.deferred,true);
    assert.equal(result.status,"PROCESSING");
    assert.equal(result.proposalStateUpdated,false);
    assert.equal(result.persistence?.success,false);
    assert.equal(result.persistence?.uncertain,true);
    console.log("COMPETING_LOSER_UNCERTAIN="+suffix);
  }else if(result.persistence?.recoveredAck===true){
    assert.equal(result.deferred,false);
    assert.equal(result.status,"FAILED");
    assert.equal(result.proposalStateUpdated,true);
    assert.equal(loseFailedAck,true,"ACK can be lost only on injected failure");
    assert.equal(result.persistence.existing,true);
    assert.equal(result.persistence.updated,false);
    console.log("COMPETING_LOSER_ACK_RECOVERED="+suffix);
  }else if(result.persistence?.existing===true){
    assert.equal(result.deferred,false);
    assert.equal(result.status,"FAILED");
    assert.equal(result.proposalStateUpdated,true);
    assert.equal(result.persistence.updated,false);
    console.log("COMPETING_LOSER_REPLAY="+suffix);
  }else{
    assert.equal(result.deferred,false);
    assert.equal(result.status,"FAILED");
    assert.equal(result.proposalStateUpdated,true);
    assert.equal(result.persistence?.updated,true);
    console.log("COMPETING_LOSER="+suffix);
  }
}
