import assert from "node:assert/strict";
import {persistExperienceSkill}
 from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {saveApprovedSkillVersion}
 from "../../backend/core/learning/approval/approvalSkillSave.js";
import {transitionCandidateFinalStatus}
 from "../../backend/core/learning/candidateMemory/candidateFinalStatus.js";
import {transitionProposalStatus}
 from "../../backend/core/learning/proposalStorage/proposalStatusCAS.js";
import {buildProposalStatusUpdate}
 from "../../backend/core/learning/proposalStorage/proposalStorageStatus.js";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";
import {candidateFinalPgClient} from "./fixtures/candidateFinalPgClient.mjs";
import {proposalStatusClient} from "./fixtures/proposalRestartPgStatusClient.mjs";

assertIsolatedCi();
const phase=process.argv[2];
assert.ok(["offline","recover","replay","race"].includes(phase));
const racing=phase==="race";
const candidate=racing
  ?"f4ca0fa8-45d0-4656-9bf6-b096a4858cc4"
  :"35b89f96-cfbf-45ab-943f-722e6b457487";
const proposal={id:racing?"ci-parallel-recovery-proposal":"ci-commit-gap-proposal",
  action:"NEW_SKILL",confidence:0.95,candidateMemory:{id:candidate},
  proposedExperience:{
    id:racing?"ci-parallel-recovery-skill":"ci-commit-gap-skill",
    name:"Crash-window Skill"
  }};
let writes=0;
const approve=async()=>{
  const receipt=await persistExperienceSkill({
    id:proposal.proposedExperience.id,name:proposal.proposedExperience.name,
    version:1,previousVersion:null,metadata:{proposalId:proposal.id}
  },{
    lookup:async id=>{
      assert.equal(id,proposal.id);
      const r=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${id}' LIMIT 1;
`);
      assert.equal(r.code,0,r.stderr);
      return r.stdout?JSON.parse(r.stdout):null;
    },
    save:async()=>{writes++;throw Error("Must never republish committed Skill");}
  });
  return {...receipt,stage:"already-saved"};
};
const result=await saveApprovedSkillVersion({
  proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.95}
},{
  approve,
  promote:({proposal:p,skillId})=>transitionCandidateFinalStatus({
    client:candidateFinalPgClient(candidate,{readbackDown:phase==="offline"}),
    id:candidate,status:"PROMOTED",
    metadata:{proposalId:p.id,skillId,resolvedAt:new Date().toISOString()}
  }),
  markApproved:p=>transitionProposalStatus({
    client:proposalStatusClient("restored",p.id),id:p.id,
    status:"APPROVED",payload:buildProposalStatusUpdate("APPROVED")
  })
});
assert.equal(writes,0);
assert.equal(result.versionCreated,false);
assert.equal(result.publicationRecovered,true);
if(phase==="offline"){
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.candidateMemory?.success,false);
  assert.equal(result.proposalStateUpdated,false);
}else{
  assert.equal(result.success,true,JSON.stringify(result));
  assert.equal(result.status,"APPROVED");
  assert.equal(result.candidateMemory?.success,true);
  if(!racing){
    assert.equal(result.candidateMemory?.updated,phase==="recover");
    assert.equal(result.persistence?.existing,phase==="replay");
  }else{
    assert.equal(result.candidateMemory?.updated===true
      ||result.candidateMemory?.existing===true,true);
    assert.equal(result.proposalStateUpdated,true);
  }
}
console.log("PASS: fresh worker recovered committed Skill gap "+phase);
