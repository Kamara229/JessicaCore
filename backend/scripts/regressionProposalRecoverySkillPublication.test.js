import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {classifyApprovalPublication}
  from "../core/learning/approval/approvalPublicationOutcome.js";

const proposal={id:"recovery-proposal",queueItemId:"recovery-queue",
  action:"NEW_SKILL",confidence:0.92,
  proposedExperience:{id:"recovery-skill",name:"Safe Skill"}};
const context={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.92}};

test("approval recovery after saved Experience and lost status ACK never reports a second version",async()=>{
  const stored=new Map();
  let published=0,updates=0,promotions=0;
  const deps={
    approve:async ({proposal:p})=>{
      const existing=stored.get(p.id);
      if(existing)return {success:true,existing:true,stage:"already-saved",
        skillId:existing.skillId,version:existing.version,
        experience:existing.experience};
      const committed={skillId:"recovery-skill",version:1,
        experience:{id:"recovery-skill",version:1}};
      stored.set(p.id,committed);
      published++;
      return {success:true,existing:false,stage:"saved",...committed};
    },
    markApproved:async()=>{
      updates++;
      return updates===1
        ? {success:false,error:"database response lost"}
        : {success:true};
    },
    promote:async()=>{
      promotions++;
      return {success:true};
    }
  };
  const first=await saveApprovedSkillVersion(context,deps);
  assert.equal(first.success,false);
  assert.equal(first.deferred,true);
  assert.equal(first.status,"PROCESSING");
  assert.equal(first.versionCreated,true);
  assert.equal(first.publicationRecovered,false);
  assert.equal(first.proposalStateUpdated,false);

  const replay=await saveApprovedSkillVersion(context,deps);
  assert.equal(replay.success,true);
  assert.equal(replay.existing,true);
  assert.equal(replay.versionCreated,false);
  assert.equal(replay.publicationRecovered,true);
  assert.equal(replay.proposalStateUpdated,true);
  assert.equal(replay.version,1);

  const duplicate=await saveApprovedSkillVersion(context,deps);
  assert.equal(duplicate.versionCreated,false);
  assert.equal(duplicate.publicationRecovered,true);
  assert.equal(published,1,"Only the original call can publish");
  assert.equal(stored.size,1);
  assert.equal(updates,3);
  assert.equal(promotions,3,"Replay may reconcile candidate state");
});

test("atomic storage existing flag is authoritative even when stage differs",async()=>{
  const deps={
    approve:async()=>({success:true,existing:true,stage:"saved",
      skillId:"recovery-skill",version:2}),
    markApproved:async()=>({success:true}),
    promote:async()=>({success:true})
  };
  const result=await saveApprovedSkillVersion(context,deps);
  assert.equal(result.success,true);
  assert.equal(result.versionCreated,false);
  assert.equal(result.publicationRecovered,true);
  assert.equal(result.version,2);
});

test("pure publication classifier recognizes pre-check and RPC recovery",()=>{
  assert.deepEqual(classifyApprovalPublication({
    existing:false,stage:"saved"
  }),{versionCreated:true,publicationRecovered:false});
  assert.deepEqual(classifyApprovalPublication({
    existing:true,stage:"saved"
  }),{versionCreated:false,publicationRecovered:true});
  assert.deepEqual(classifyApprovalPublication({
    existing:true,stage:"already-saved"
  }),{versionCreated:false,publicationRecovered:true});
});
