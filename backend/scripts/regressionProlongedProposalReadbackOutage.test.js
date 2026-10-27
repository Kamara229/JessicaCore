import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {transitionProposalStatus}
  from "../core/learning/proposalStorage/proposalStatusCAS.js";
import {buildProposalStatusUpdate}
  from "../core/learning/proposalStorage/proposalStorageStatus.js";
import {finalStatusReadbackOutageStore}
  from "./fixtures/finalStatusReadbackOutageStore.js";

const proposal={
  id:"outage-proposal",action:"NEW_SKILL",confidence:0.94,
  candidateMemory:{id:"00000000-0000-4000-8000-000000009441"},
  proposedExperience:{id:"outage-skill",name:"Durably saved Skill"}
};
const input={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.94}};

function newWorker(store){
  return {
    ...store.createWorker(),
    markApproved:async current=>transitionProposalStatus({
      client:store.client,id:current.id,status:"APPROVED",
      payload:buildProposalStatusUpdate("APPROVED")
    })
  };
}

test("8 readback failures after committed APPROVED never republish or claim success",async()=>{
  const db=finalStatusReadbackOutageStore({
    postCommitAckLoss:true,readbackOutage:true
  });
  const first=await saveApprovedSkillVersion(input,newWorker(db));
  assert.equal(first.success,false);
  assert.equal(first.deferred,true);
  assert.equal(first.persistence.uncertain,true);
  assert.equal(first.status,"PROCESSING");
  assert.equal(db.row.status,"APPROVED");
  const approvedAt=db.row.approved_at;

  for(let attempt=0;attempt<8;attempt++){
    // A newly constructed Worker can use the same persistent database.
    const replay=await saveApprovedSkillVersion(input,newWorker(db));
    assert.equal(replay.success,false);
    assert.equal(replay.deferred,true);
    assert.equal(replay.proposalStateUpdated,false);
    assert.equal(replay.persistence.uncertain,true);
    assert.equal(replay.status,"PROCESSING");
    assert.equal(db.row.approved_at,approvedAt);
  }
  assert.equal(db.state.statusWrites,1);
  assert.equal(db.state.versions.size,1);
  assert.equal(db.state.versionWrites,0);
  assert.equal(db.state.candidate.revision,1);
  assert.equal(db.state.candidateWrites,0);
  assert.equal(db.state.readbackFailures,9);

  db.setReadbackAvailable(true);
  const resumed=await saveApprovedSkillVersion(input,newWorker(db));
  assert.equal(resumed.success,true);
  assert.equal(resumed.deferred,false);
  assert.equal(resumed.status,"APPROVED");
  assert.equal(resumed.existing,true);
  assert.equal(resumed.publicationRecovered,true);
  assert.equal(resumed.versionCreated,false);
  assert.equal(resumed.persistence.existing,true);
  assert.equal(resumed.persistence.updated,false);
  assert.equal(db.state.statusWrites,1);
  assert.equal(db.row.approved_at,approvedAt);
  assert.equal(db.state.versions.size,1);
  assert.equal(db.state.candidate.revision,1);
});

test("outage before status COMMIT stays unconfirmed then finalizes exactly once",async()=>{
  const db=finalStatusReadbackOutageStore({
    updateOutage:true,readbackOutage:true
  });
  for(let attempt=0;attempt<7;attempt++){
    const result=await saveApprovedSkillVersion(input,newWorker(db));
    assert.equal(result.success,false);
    assert.equal(result.deferred,true);
    assert.equal(result.persistence.uncertain,true);
    assert.equal(result.status,"PROCESSING");
    assert.equal(db.row.status,"PROCESSING");
    assert.equal(db.state.statusWrites,0);
  }
  db.setUpdateAvailable(true);
  // Successful SELECT from UPDATE does not require the readback channel.
  const result=await saveApprovedSkillVersion(input,newWorker(db));
  assert.equal(result.success,true);
  assert.equal(result.status,"APPROVED");
  assert.equal(result.persistence.existing,false);
  const committedAt=db.row.approved_at;
  assert.equal(db.state.statusWrites,1);
  assert.equal(db.state.versions.size,1);
  db.setReadbackAvailable(true);
  const repeated=await saveApprovedSkillVersion(input,newWorker(db));
  assert.equal(repeated.success,true);
  assert.equal(repeated.persistence.existing,true);
  assert.equal(db.row.approved_at,committedAt);
  assert.equal(db.state.statusWrites,1);
  assert.equal(db.state.candidate.revision,1);
  assert.equal(db.state.candidateWrites,0);
});

test("concurrent retries while readback is down cannot overwrite approved timestamp",async()=>{
  const db=finalStatusReadbackOutageStore({
    postCommitAckLoss:true,readbackOutage:true
  });
  await saveApprovedSkillVersion(input,newWorker(db));
  const approvedAt=db.row.approved_at;
  const results=await Promise.all(Array.from({length:12},
    ()=>saveApprovedSkillVersion(input,newWorker(db))));
  assert.ok(results.every(r=>r.success===false&&r.deferred===true));
  assert.ok(results.every(r=>r.persistence?.uncertain===true));
  assert.equal(db.row.approved_at,approvedAt);
  assert.equal(db.state.statusWrites,1);
  assert.equal(db.state.versions.size,1);
  assert.equal(db.state.candidate.revision,1);
  db.setReadbackAvailable(true);
  const final=await saveApprovedSkillVersion(input,newWorker(db));
  assert.equal(final.success,true);
  assert.equal(final.status,"APPROVED");
  assert.equal(db.state.statusWrites,1);
});
