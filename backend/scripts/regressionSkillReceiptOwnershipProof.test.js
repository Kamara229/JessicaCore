import test from "node:test";
import assert from "node:assert/strict";
import {approveAndSaveLearningProposal}
  from "../experience/learning/learningApproval.js";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";

const skillId="receipt-identity-verified-skill";
const proposal={
  id:"receipt-identity-verified-proposal",action:"NEW_SKILL",
  confidence:0.92,
  candidateMemory:{id:"00000000-0000-4000-8000-000000000913"},
  proposedExperience:{id:skillId,name:"Existing verified Skill"}
};
const input={proposal,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.92}};
const good={
  skillId,version:1,
  payload:{id:skillId,version:1,metadata:{proposalId:proposal.id}}
};

function deps(readReceipt){
  const events={reads:0,history:0,writes:0,promotions:0,
    approvals:0,failures:0,conflictChecks:0};
  return {
    events,
    approve:({proposal:p,autonomy,confidence})=>
      approveAndSaveLearningProposal({proposal:p,autonomy,confidence},{
        lookupReceipt:async id=>{
          events.reads++;
          assert.equal(id,proposal.id);
          return readReceipt();
        },
        loadHistory:async()=>{events.history++;throw Error("Forbidden History read");},
        saveSkill:async()=>{events.writes++;throw Error("Forbidden Skill write");}
      }),
    promote:async()=>{events.promotions++;return {success:true,existing:true};},
    markApproved:async()=>{events.approvals++;return {success:true};},
    markFailed:async()=>{events.failures++;throw Error("No FAILED writes allowed");},
    checkPublicationConflict:async()=>{
      events.conflictChecks++;
      return {confirmed:false,checked:false};
    }
  };
}

for(const [name,receipt] of [
  ["wrong Skill ID",{...good,skillId:"unrelated-skill"}],
  ["wrong payload Skill ID",{...good,payload:{...good.payload,id:"wrong-skill"}}],
  ["foreign Proposal owner",{...good,payload:{
    ...good.payload,metadata:{proposalId:"other-proposal"}}}],
  ["missing immutable payload",{...good,payload:null}],
  ["invalid immutable version",{...good,version:0}],
  ["noninteger immutable version",{...good,version:"1"}],
]){
  test("receipt readback "+name+" cannot promote Candidate or finalize Proposal",async()=>{
    let current=structuredClone(receipt);
    const d=deps(()=>current);
    for(let i=0;i<2;i++){
      const result=await saveApprovedSkillVersion(input,d);
      assert.equal(result.success,false);
      assert.equal(result.deferred,true);
      assert.equal(result.uncertain,true);
      assert.equal(result.stage,"idempotency");
      assert.equal(result.status,"PROCESSING");
      assert.equal(result.proposalStateUpdated,false);
      assert.equal(result.candidateMemoryStateUpdated,false);
      assert.match(result.error,/receipt/i);
    }
    assert.equal(d.events.writes,0);
    assert.equal(d.events.history,0);
    assert.equal(d.events.promotions,0);
    assert.equal(d.events.approvals,0);
    assert.equal(d.events.failures,0);
    current=structuredClone(good);
    const verified=await saveApprovedSkillVersion(input,d);
    assert.equal(verified.success,true);
    assert.equal(verified.existing,true);
    assert.equal(verified.versionCreated,false);
    assert.equal(verified.publicationRecovered,true);
    assert.equal(verified.status,"APPROVED");
    assert.equal(d.events.writes,0);
    assert.equal(d.events.promotions,1);
    assert.equal(d.events.approvals,1);
  });
}

test("receipt read errors are deferred and restored by matching receipt",async()=>{
  let outage=true;
  const d=deps(()=>{
    if(outage)throw Error("Receipt hydration timeout");
    return structuredClone(good);
  });
  for(let i=0;i<7;i++){
    const result=await saveApprovedSkillVersion(input,d);
    assert.equal(result.success,false);
    assert.equal(result.uncertain,true);
    assert.equal(result.status,"PROCESSING");
  }
  outage=false;
  const recovered=await saveApprovedSkillVersion(input,d);
  assert.equal(recovered.success,true);
  assert.equal(recovered.existing,true);
  assert.equal(d.events.writes,0);
  assert.equal(d.events.failures,0);
  assert.equal(d.events.promotions,1);
  assert.equal(d.events.approvals,1);
});
