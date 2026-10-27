import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {transitionCandidateFinalStatus}
  from "../core/learning/candidateMemory/candidateFinalStatus.js";
import {candidatePromotionRecoveryStore}
  from "./fixtures/candidatePromotionRecoveryStore.js";

const proposal={id:"resume-after-publication-proposal",action:"NEW_SKILL",
  confidence:0.93,
  candidateMemory:{id:"00000000-0000-4000-8000-000000000701"},
  proposedExperience:{id:"resume-after-publication-skill",name:"Existing Skill"}};
const input={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.93}};

// Represents immutable history already committed before process termination.
// The fresh Approval worker must consume the receipt, never republish it.
function resumedWorker(db){
  return saveApprovedSkillVersion(input,{
    approve:async()=>({success:true,existing:true,
      stage:"already-saved",skillId:proposal.proposedExperience.id,
      version:1,experience:{id:proposal.proposedExperience.id,version:1}}),
    promote:async({proposal:p,skillId})=>
      transitionCandidateFinalStatus({
        client:db.client,id:p.candidateMemory.id,status:"PROMOTED",
        metadata:{proposalId:p.id,skillId,resolvedAt:new Date().toISOString()}
      }),
    markApproved:async()=>{
      db.state.proposalWrites++;
      db.state.proposalStatus="APPROVED";
      return {success:true};
    }
  });
}

test("fresh worker finishes already-published Skill whose Candidate is ACTIVE",async()=>{
  const db=candidatePromotionRecoveryStore();
  db.state.versions=1; // The immutable publication predates this process.
  const result=await resumedWorker(db);
  assert.equal(result.success,true);
  assert.equal(result.status,"APPROVED");
  assert.equal(result.publicationRecovered,true);
  assert.equal(result.versionCreated,false);
  assert.equal(result.existing,true);
  assert.equal(db.state.candidate.status,"PROMOTED");
  assert.equal(db.state.candidate.revision,1);
  assert.equal(db.state.candidateWrites,1);
  assert.equal(db.state.proposalWrites,1);
  assert.equal(db.state.versions,1);
});

test("fresh worker finalizes when Skill and Candidate committed but Approval ACK was lost",async()=>{
  const db=candidatePromotionRecoveryStore();
  db.state.versions=1;
  db.state.candidate.status="PROMOTED";
  db.state.candidate.revision=1;
  db.state.candidate.metadata={
    proposalId:proposal.id,skillId:proposal.proposedExperience.id
  };
  const result=await resumedWorker(db);
  assert.equal(result.success,true);
  assert.equal(result.status,"APPROVED");
  assert.equal(result.candidateMemory.existing,true);
  assert.equal(result.publicationRecovered,true);
  assert.equal(result.versionCreated,false);
  assert.equal(db.state.candidateWrites,0);
  assert.equal(db.state.proposalWrites,1);
  assert.equal(db.state.versions,1);
});

test("another Proposal's PROMOTED Candidate is never marked approved",async()=>{
  const db=candidatePromotionRecoveryStore();
  db.state.versions=1;
  db.state.candidate.status="PROMOTED";
  db.state.candidate.revision=1;
  db.state.candidate.metadata={
    proposalId:"different-proposal",skillId:"different-skill"
  };
  const result=await resumedWorker(db);
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(db.state.candidateWrites,0);
  assert.equal(db.state.proposalWrites,0);
  assert.equal(db.state.versions,1);
});
