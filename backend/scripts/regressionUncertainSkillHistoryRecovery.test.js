import test from "node:test";
import assert from "node:assert/strict";
import {approveAndSaveLearningProposal}
  from "../experience/learning/learningApproval.js";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";

const skillId="history-outage-skill";
const proposal={
  id:"history-outage-proposal",action:"NEW_SKILL",confidence:0.95,
  candidateMemory:{id:"00000000-0000-4000-8000-000000001101"},
  proposedExperience:{id:skillId,name:"History consistency regression"}
};
const context={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.95}};
const receipt={skillId,version:1,payload:{
  id:skillId,version:1,metadata:{proposalId:proposal.id}
}};

function harness(initialHistory){
  let currentHistory=initialHistory,existingReceipt=null;
  const state={historyReads:0,receiptReads:0,skillWrites:0,
    candidateWrites:0,approvedWrites:0,failedWrites:0,conflictReads:0};
  const deps={
    approve:({proposal:p,autonomy,confidence})=>
      approveAndSaveLearningProposal({proposal:p,autonomy,confidence},{
        lookupReceipt:async()=>{
          state.receiptReads++;
          return existingReceipt&&structuredClone(existingReceipt);
        },
        loadHistory:async()=>{
          state.historyReads++;
          if(currentHistory instanceof Error)throw currentHistory;
          return structuredClone(currentHistory);
        },
        saveSkill:async()=>{state.skillWrites++;throw Error("No unverified Skill write");}
      }),
    promote:async()=>{state.candidateWrites++;return {success:true,existing:true};},
    markApproved:async()=>{state.approvedWrites++;return {success:true};},
    markFailed:async()=>{state.failedWrites++;throw Error("Invalid terminal FAILED");},
    checkPublicationConflict:async()=>{
      state.conflictReads++;return {confirmed:false,checked:false};
    }
  };
  return {deps,state,
    restoreReceipt(){existingReceipt=receipt;},
    setHistory(next){currentHistory=next;}
  };
}

for(const [label,badHistory] of [
  ["history load timeout",new Error("history query timed out")],
  ["null response",null],
  ["non-array response",{rows:[]}],
  ["duplicate versions",[{version:1},{version:1}]],
  ["missing v2",[{version:1},{version:3}]],
  ["no initial version",[{version:2}]],
  ["invalid version",[{version:0}]],
  ["malformed item",[null]],
  ["non-integer version",[{version:1.5}]]
]){
  test("unverified "+label+" defers Approval until immutable receipt returns",async()=>{
    const h=harness(badHistory);
    for(let i=0;i<3;i++){
      const r=await saveApprovedSkillVersion(context,h.deps);
      assert.equal(r.success,false);
      assert.equal(r.deferred,true);
      assert.equal(r.uncertain,true);
      assert.equal(r.status,"PROCESSING");
      assert.equal(r.stage,"history");
      assert.equal(r.proposalStateUpdated,false);
      assert.equal(r.candidateMemoryStateUpdated,false);
    }
    assert.equal(h.state.historyReads,3);
    assert.equal(h.state.receiptReads,3);
    assert.equal(h.state.skillWrites,0);
    assert.equal(h.state.candidateWrites,0);
    assert.equal(h.state.approvedWrites,0);
    assert.equal(h.state.failedWrites,0);
    h.restoreReceipt();
    const recovered=await saveApprovedSkillVersion(context,h.deps);
    assert.equal(recovered.success,true);
    assert.equal(recovered.status,"APPROVED");
    assert.equal(recovered.publicationRecovered,true);
    assert.equal(recovered.versionCreated,false);
    assert.equal(recovered.existing,true);
    assert.equal(h.state.historyReads,3);
    assert.equal(h.state.skillWrites,0);
    assert.equal(h.state.candidateWrites,1);
    assert.equal(h.state.approvedWrites,1);
    assert.equal(h.state.failedWrites,0);
  });
}

test("12 parallel incomplete History reads never publish or finalize",async()=>{
  const h=harness([{version:2}]);
  const results=await Promise.all(Array.from({length:12},
    ()=>saveApprovedSkillVersion(context,h.deps)));
  assert.ok(results.every(r=>r.success===false&&r.deferred===true
    &&r.uncertain===true&&r.status==="PROCESSING"));
  assert.equal(h.state.historyReads,12);
  assert.equal(h.state.skillWrites,0);
  assert.equal(h.state.candidateWrites,0);
  assert.equal(h.state.approvedWrites,0);
  assert.equal(h.state.failedWrites,0);
});
