import assert from "node:assert/strict";
import {persistExperienceSkill}
 from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {saveApprovedSkillVersion}
 from "../../backend/core/learning/approval/approvalSkillSave.js";
import {transitionProposalStatus}
 from "../../backend/core/learning/proposalStorage/proposalStatusCAS.js";
import {buildProposalStatusUpdate}
 from "../../backend/core/learning/proposalStorage/proposalStorageStatus.js";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";
import {proposalStatusClient} from "./fixtures/proposalRestartPgStatusClient.mjs";

assertIsolatedCi();
const phase=process.argv[2];
assert.ok(["lost-ack","down","restored"].includes(phase));
const candidate="fcdf796b-1f12-4330-9be2-5c642dfbf356";
const proposal={id:"ci-restarted-proposal",action:"NEW_SKILL",confidence:0.94,
  candidateMemory:{id:candidate},
  proposedExperience:{id:"ci-restarted-skill",name:"Committed restarted Skill"}};
const skill={id:"ci-restarted-skill",name:"Committed restarted Skill",
  version:1,previousVersion:null,metadata:{proposalId:proposal.id}};

const lookup=async id=>{
  assert.equal(id,proposal.id);
  const r=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${id}' LIMIT 1;
`);
  assert.equal(r.code,0,r.stderr);
  return r.stdout?JSON.parse(r.stdout):null;
};
const deps={
  approve:async()=>{
    const receipt=await persistExperienceSkill(skill,{
      lookup,save:async()=>{throw Error("Restart must never republish Skill");}
    });
    return {...receipt,stage:"already-saved"};
  },
  promote:async({proposal:p,skillId})=>{
    const r=await sql(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM public.learning_candidates
 WHERE id='${candidate}' AND status='PROMOTED'
 AND metadata->>'proposalId'='${p.id}'
 AND metadata->>'skillId'='${skillId}') THEN 1 ELSE 0 END;
`);
    assert.equal(r.code,0,r.stderr);
    return r.stdout==="1"
      ? {success:true,existing:true,updated:false}
      : {success:false,error:"Candidate not durably promoted"};
  },
  markApproved:async p=>transitionProposalStatus({
    client:proposalStatusClient(phase),id:p.id,status:"APPROVED",
    payload:buildProposalStatusUpdate("APPROVED")
  })
};
const result=await saveApprovedSkillVersion({
  proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.94}
},deps);

assert.equal(result.versionCreated,false);
assert.equal(result.publicationRecovered,true);
if(phase==="restored"){
  assert.equal(result.success,true,JSON.stringify(result));
  assert.equal(result.deferred,false);
  assert.equal(result.status,"APPROVED");
  assert.equal(result.persistence.existing,true);
  assert.equal(result.persistence.updated,false);
}else{
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.persistence?.uncertain,true);
  assert.equal(result.proposalStateUpdated,false);
}
console.log("PASS: fresh Approval worker process "+phase);
