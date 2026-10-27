import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {transitionProposalStatus}
  from "../core/learning/proposalStorage/proposalStatusCAS.js";
import {buildProposalStatusUpdate}
  from "../core/learning/proposalStorage/proposalStorageStatus.js";

const proposal={id:"final-ack-loss-proposal",action:"NEW_SKILL",
  confidence:0.93,
  candidateMemory:{id:"00000000-0000-4000-8000-000000000893"},
  proposedExperience:{id:"final-ack-loss-skill",name:"Existing published Skill"}};
const context={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.93}};

function finalStatusStore({loseAck=false,loseReadback=false,preCommitFailure=false}={}){
  let proposalRow={id:proposal.id,status:"PROCESSING",approved_at:null,
    rejected_at:null,processing_at:"2026-10-27T16:00:00.000Z"};
  let candidate={status:"PROMOTED",proposalId:proposal.id,
    skillId:proposal.proposedExperience.id};
  let statusWrites=0,versionWrites=0,promotionWrites=0;
  let dropAck=loseAck,dropReadback=loseReadback,failBeforeCommit=preCommitFailure;
  const versions=new Map();
  const client={from(table){
    assert.equal(table,"learning_proposals");
    return {payload:null,predicates:{},
      update(value){this.payload=value;return this;},
      select(){return this;},
      eq(k,v){this.predicates[k]=v;return this;},
      async maybeSingle(){
        const found=Object.entries(this.predicates)
          .every(([k,v])=>proposalRow?.[k]===v);
        if(!this.payload){
          if(dropReadback){dropReadback=false;
            return {data:null,error:{message:"post-commit readback unavailable"}};
          }
          return {data:found?structuredClone(proposalRow):null,error:null};
        }
        if(!found)return {data:null,error:null};
        if(failBeforeCommit){failBeforeCommit=false;
          return {data:null,error:{message:"pre-commit connection dropped"}};
        }
        proposalRow={...proposalRow,...this.payload};
        statusWrites++;
        if(dropAck){dropAck=false;
          return {data:null,error:{message:"APPROVED ACK lost after COMMIT"}};
        }
        return {data:structuredClone(proposalRow),error:null};
      }};
  }};
  const deps={
    approve:async({proposal:p})=>{
      assert.equal(p.id,proposal.id);
      const existing=versions.has(p.id);
      if(!existing){
        versions.set(p.id,{skillId:proposal.proposedExperience.id,version:1});
        versionWrites++;
      }
      return {success:true,existing,
        stage:existing?"already-saved":"saved",version:1,
        skillId:proposal.proposedExperience.id,
        experience:{id:proposal.proposedExperience.id,version:1}};
    },
    promote:async({proposal:p,skillId})=>{
      assert.equal(p.id,candidate.proposalId);
      assert.equal(skillId,candidate.skillId);
      assert.equal(candidate.status,"PROMOTED");
      return {success:true,existing:true,updated:false};
    },
    markApproved:async p=>transitionProposalStatus({
      client,id:p.id,status:"APPROVED",
      payload:buildProposalStatusUpdate("APPROVED")
    })
  };
  return {deps,client,
    get row(){return structuredClone(proposalRow);},
    get statusWrites(){return statusWrites;},
    get versionWrites(){return versionWrites;},
    get promotionWrites(){return promotionWrites;},
    get candidate(){return candidate;}
  };
}

test("post-COMMIT APPROVED ACK loss is confirmed from durable readback",async()=>{
  const db=finalStatusStore({loseAck:true});
  const first=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(first.success,true);
  assert.equal(first.deferred,false);
  assert.equal(first.status,"APPROVED");
  assert.equal(first.persistence.recoveredAck,true);
  assert.equal(first.persistence.existing,true);
  assert.equal(first.proposalStateUpdated,true);
  assert.equal(first.versionCreated,true);
  assert.equal(db.row.status,"APPROVED");
  const approvedAt=db.row.approved_at;
  const replay=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(replay.success,true);
  assert.equal(replay.existing,true);
  assert.equal(replay.publicationRecovered,true);
  assert.equal(replay.versionCreated,false);
  assert.equal(replay.persistence.existing,true);
  assert.equal(db.row.approved_at,approvedAt);
  assert.equal(db.statusWrites,1);
  assert.equal(db.versionWrites,1);
  assert.equal(db.promotionWrites,0);
});

test("lost ACK and unavailable readback defer, then replay confirms terminal state",async()=>{
  const db=finalStatusStore({loseAck:true,loseReadback:true});
  const interrupted=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(interrupted.success,false);
  assert.equal(interrupted.deferred,true);
  assert.equal(interrupted.status,"PROCESSING");
  assert.equal(interrupted.proposalStateUpdated,false);
  assert.equal(interrupted.persistence.uncertain,true);
  // The actual database committed APPROVED before its ACK was lost.
  assert.equal(db.row.status,"APPROVED");
  assert.equal(db.statusWrites,1);
  const committedAt=db.row.approved_at;
  const resumed=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(resumed.success,true);
  assert.equal(resumed.status,"APPROVED");
  assert.equal(resumed.publicationRecovered,true);
  assert.equal(resumed.versionCreated,false);
  assert.equal(resumed.persistence.existing,true);
  assert.equal(db.statusWrites,1,"Replay cannot rewrite terminal Proposal");
  assert.equal(db.row.approved_at,committedAt);
  assert.equal(db.versionWrites,1);
});

test("pre-COMMIT status error with unconfirmed readback does not claim approval",async()=>{
  const db=finalStatusStore({preCommitFailure:true,loseReadback:true});
  const first=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(first.success,false);
  assert.equal(first.deferred,true);
  assert.equal(db.row.status,"PROCESSING");
  assert.equal(db.statusWrites,0);
  const retry=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(retry.success,true);
  assert.equal(retry.status,"APPROVED");
  assert.equal(db.row.status,"APPROVED");
  assert.equal(db.statusWrites,1);
  assert.equal(db.versionWrites,1);
});

test("an already-terminal foreign decision is never overwritten as APPROVED",async()=>{
  const db=finalStatusStore();
  db.deps.markApproved=async p=>{
    const terminal=await transitionProposalStatus({
      client:db.client,id:p.id,status:"REJECTED",
      payload:buildProposalStatusUpdate("REJECTED")
    });
    assert.equal(terminal.success,true);
    return transitionProposalStatus({
      client:db.client,id:p.id,status:"APPROVED",
      payload:buildProposalStatusUpdate("APPROVED")
    });
  };
  const result=await saveApprovedSkillVersion(context,db.deps);
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.proposalStateUpdated,false);
  assert.equal(db.row.status,"REJECTED");
  assert.equal(db.statusWrites,1);
  assert.equal(db.versionWrites,1);
});
