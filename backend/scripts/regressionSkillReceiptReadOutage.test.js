import test from "node:test";
import assert from "node:assert/strict";
import {approveAndSaveLearningProposal}
  from "../experience/learning/learningApproval.js";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";

const proposal={
  id:"receipt-read-outage-proposal",action:"NEW_SKILL",
  confidence:0.94,
  candidateMemory:{id:"00000000-0000-4000-8000-000000008412"},
  proposedExperience:{
    id:"receipt-read-outage-skill",name:"Durably saved Skill",
    workflow:["Verify immutable publication"],examples:[{
      task:"Prove exactly one committed Skill",success:true,status:"COMPLETED",
      result:{status:"COMPLETED",success:true,verified:true}
    }],learning:{confidence:0.94,maturity:0.91,occurrences:3,successRate:1}
  }
};
const receipt={
  skillId:proposal.proposedExperience.id,version:1,
  payload:{id:proposal.proposedExperience.id,version:1,
    metadata:{proposalId:proposal.id}}
};
const context={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.94}};

function harness({readReceipt,readHistory=async()=>{throw Error("History must not be loaded");}}={}){
  const state={skillWrites:0,candidateWrites:0,proposalWrites:0,
    conflictChecks:0,failedWrites:0};
  const approve=({proposal:p,autonomy,confidence})=>
    approveAndSaveLearningProposal({proposal:p,autonomy,confidence},{
      lookupReceipt:readReceipt,loadHistory:readHistory,
      saveSkill:async()=>{state.skillWrites++;throw Error("Must not republish");}
    });
  const deps={
    approve,
    promote:async()=>{state.candidateWrites++;return {success:true,existing:true};},
    markApproved:async()=>{state.proposalWrites++;return {success:true};},
    markFailed:async()=>{state.failedWrites++;throw Error("Must never mark FAILED");},
    checkPublicationConflict:async()=>{
      state.conflictChecks++;return {confirmed:false,checked:false};
    }
  };
  return {state,deps};
}

test("8 failed immutable Skill receipt reads keep Approval PROCESSING, then recover",async()=>{
  let outage=true,reads=0;
  const h=harness({readReceipt:async()=>{
    reads++;
    if(outage)throw Error("Supabase receipt lookup timeout");
    return structuredClone(receipt);
  }});
  for(let i=0;i<8;i++){
    const res=await saveApprovedSkillVersion(context,h.deps);
    assert.equal(res.success,false);
    assert.equal(res.deferred,true);
    assert.equal(res.uncertain,true);
    assert.equal(res.status,"PROCESSING");
    assert.equal(res.stage,"idempotency");
    assert.equal(res.proposalStateUpdated,false);
    assert.match(res.error,/receipt lookup timeout/);
    assert.equal(h.state.failedWrites,0);
  }
  assert.equal(h.state.conflictChecks,8);
  assert.equal(h.state.skillWrites,0);
  assert.equal(h.state.candidateWrites,0);
  assert.equal(h.state.proposalWrites,0);
  outage=false;
  const result=await saveApprovedSkillVersion(context,h.deps);
  assert.equal(result.success,true);
  assert.equal(result.status,"APPROVED");
  assert.equal(result.existing,true);
  assert.equal(result.versionCreated,false);
  assert.equal(result.publicationRecovered,true);
  assert.equal(result.version,1);
  assert.equal(h.state.skillWrites,0);
  assert.equal(h.state.candidateWrites,1);
  assert.equal(h.state.proposalWrites,1);
  assert.equal(h.state.failedWrites,0);
  assert.equal(reads,9);
});

test("unavailable Skill history after absent preflight receipt also defers",async()=>{
  let lookupCalls=0,historyCalls=0;
  const h=harness({
    readReceipt:async()=>{lookupCalls++;return null;},
    readHistory:async()=>{historyCalls++;throw Error("history API unavailable");}
  });
  const result=await saveApprovedSkillVersion(context,h.deps);
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.uncertain,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.stage,"history");
  assert.match(result.error,/history API unavailable/);
  assert.equal(lookupCalls,1);
  assert.equal(historyCalls,1);
  assert.equal(h.state.skillWrites,0);
  assert.equal(h.state.failedWrites,0);
  assert.equal(h.state.candidateWrites,0);
  assert.equal(h.state.proposalWrites,0);
});

test("parallel receipt outages cannot publish or terminally fail Proposal",async()=>{
  let reads=0;
  const h=harness({readReceipt:async()=>{
    reads++;throw Error("database readback down");
  }});
  const results=await Promise.all(Array.from({length:12},
    ()=>saveApprovedSkillVersion(context,h.deps)));
  assert.equal(reads,12);
  assert.ok(results.every(r=>r.uncertain===true && r.deferred===true &&
    r.status==="PROCESSING" && r.success===false));
  assert.equal(h.state.skillWrites,0);
  assert.equal(h.state.candidateWrites,0);
  assert.equal(h.state.proposalWrites,0);
  assert.equal(h.state.failedWrites,0);
  assert.equal(h.state.conflictChecks,12);
});

test("receipt load errors carry explicit uncertainty before any Skill write",async()=>{
  let writes=0;
  const result=await approveAndSaveLearningProposal({
    proposal,autonomy:context.autonomy
  },{
    lookupReceipt:async()=>{throw Error("receipt hydration unavailable");},
    saveSkill:async()=>{writes++;}
  });
  assert.equal(result.success,false);
  assert.equal(result.uncertain,true);
  assert.equal(result.deferred,true);
  assert.equal(result.stage,"idempotency");
  assert.equal(writes,0);
});
