import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion} from "../core/learning/approval/approvalSkillSave.js";
import {transitionCandidateFinalStatus} from "../core/learning/candidateMemory/candidateFinalStatus.js";
import {candidatePromotionRecoveryStore} from "./fixtures/candidatePromotionRecoveryStore.js";

const proposal={id:"candidate-replay-proposal",confidence:0.91,
  candidateMemory:{id:"00000000-0000-4000-8000-000000000701"},
  proposedExperience:{id:"candidate-replay-skill",name:"Verified"}};
const context={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.91}};

function dependencies(db,{candidateAckLost=false,proposalAckLost=false}={}){
  db.state.commitAcknowledgmentLost=candidateAckLost;
  let loseProposalAck=proposalAckLost;
  return {
    approve:async()=>{
      const existing=db.state.versions===1;
      if(!existing)db.state.versions++;
      return {success:true,existing,stage:existing?"already-saved":"saved",
        skillId:"candidate-replay-skill",version:1,
        experience:{id:"candidate-replay-skill",version:1}};
    },
    promote:async({proposal:p,skillId})=>
      transitionCandidateFinalStatus({
        client:db.client,id:p.candidateMemory.id,status:"PROMOTED",
        metadata:{proposalId:p.id,skillId,resolvedAt:new Date().toISOString()}
      }),
    markApproved:async()=>{
      db.state.proposalWrites++;
      if(loseProposalAck){
        loseProposalAck=false;
        return {success:false,error:"Proposal ACK lost"};
      }
      db.state.proposalStatus="APPROVED";
      return {success:true};
    }
  };
}

test("lost Candidate COMMIT ACK remains recoverable; replay promotes once",async()=>{
  const db=candidatePromotionRecoveryStore();
  const deps=dependencies(db,{candidateAckLost:true});
  const first=await saveApprovedSkillVersion(context,deps);
  assert.equal(first.success,false);
  assert.equal(first.deferred,true);
  assert.equal(first.status,"PROCESSING");
  assert.equal(first.proposalStateUpdated,false);
  assert.equal(db.state.candidate.status,"PROMOTED");
  assert.equal(db.state.proposalWrites,0);
  assert.equal(db.state.candidateWrites,1);
  assert.equal(db.state.versions,1);
  const retry=await saveApprovedSkillVersion(context,deps);
  assert.equal(retry.success,true);
  assert.equal(retry.status,"APPROVED");
  assert.equal(retry.publicationRecovered,true);
  assert.equal(retry.versionCreated,false);
  assert.equal(retry.candidateMemory.existing,true);
  assert.equal(db.state.candidateWrites,1);
  assert.equal(db.state.proposalWrites,1);
  assert.equal(db.state.versions,1);
});

test("failed Candidate update before COMMIT is retried before final Approval",async()=>{
  const db=candidatePromotionRecoveryStore();
  const deps=dependencies(db);
  const promote=deps.promote;
  let outage=true;
  deps.promote=async args=>{
    if(outage){outage=false;return {success:false,error:"database outage"};}
    return promote(args);
  };
  const first=await saveApprovedSkillVersion(context,deps);
  assert.equal(first.deferred,true);
  assert.equal(db.state.proposalWrites,0);
  assert.equal(db.state.candidate.status,"ACTIVE");
  const retry=await saveApprovedSkillVersion(context,deps);
  assert.equal(retry.success,true);
  assert.equal(db.state.candidate.status,"PROMOTED");
  assert.equal(db.state.candidateWrites,1);
  assert.equal(db.state.proposalWrites,1);
  assert.equal(db.state.versions,1);
});
