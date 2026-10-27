import test from "node:test";
import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../experience/storage/experiencePersistence/experienceWriter.js";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {recoverExperienceSaveReceipt}
  from "../experience/storage/experiencePersistence/experienceAckRecovery.js";

const proposalId="ack-loss-proposal";
const experience={id:"ack-loss-skill",name:"Verified skill",version:1,
  previousVersion:null,workflow:["Verify source"],metadata:{proposalId}};
const committed={skillId:experience.id,version:1,payload:structuredClone(experience)};
function deps(mode="throw"){
  let writes=0;
  let row=null;
  const lookup=async id=>{
    assert.equal(id,proposalId);
    return row&&structuredClone(row);
  };
  const save=async value=>{
    writes++;
    row={...committed,payload:structuredClone(value)};
    if(mode==="throw")throw Error("network disconnected after commit");
    return {success:false,error:"response missing"};
  };
  return {lookup,save,get writes(){return writes;}};
}

for(const mode of ["throw","result-false"]){
  test("lost Experience save ACK ("+mode+") is recovered by immutable Proposal receipt",async()=>{
    const mock=deps(mode);
    const result=await persistExperienceSkill(experience,mock);
    assert.equal(result.success,true);
    assert.equal(result.existing,true);
    assert.equal(result.commitRecovered,true);
    assert.equal(result.version,1);
    assert.equal(result.skillId,experience.id);
    assert.equal(mock.writes,1,"Recovery must never repeat a write");
    const replay=await persistExperienceSkill(experience,mock);
    assert.equal(replay.success,true);
    assert.equal(replay.existing,true);
    assert.equal(mock.writes,1,"Durable replay must not publish a second version");
  });
}

test("missing readback after ambiguous commit fails closed without re-saving",async()=>{
  const mock={lookup:async()=>null,save:async()=>{throw Error("ACK lost");}};
  const result=await persistExperienceSkill(experience,mock);
  assert.equal(result.success,false);
  assert.equal(result.uncertain,true);
  assert.match(result.error,/ACK lost/);
});

test("unavailable readback preserves uncertainty rather than false success",async()=>{
  const result=await persistExperienceSkill(experience,{
    lookup:async()=>{throw Error("database offline");},
    save:async()=>{throw Error("response lost");}
  });
  assert.equal(result.success,false);
  assert.equal(result.uncertain,true);
  assert.match(result.error,/database offline/);
});

test("receipt mismatching Proposal or Skill cannot count as recovered commit",async()=>{
  for(const saved of [
    {...committed,skillId:"other-skill"},
    {...committed,payload:{...committed.payload,metadata:{proposalId:"other"}}},
    {...committed,version:0}
  ]){
    const result=await recoverExperienceSaveReceipt({
      proposalId,skillId:experience.id,reason:"ACK lost",
      lookup:async()=>saved
    });
    assert.equal(result.success,false);
    assert.equal(result.uncertain,true);
  }
});

test("unconfirmed commit leaves Proposal PROCESSING for recovery; no FAILED write",async()=>{
  let sideEffects=0;
  const result=await saveApprovedSkillVersion({
    proposal:{id:proposalId,confidence:0.9},
    review:{valid:true},quality:{passed:true},
    autonomy:{action:"AUTO_APPROVE",confidence:0.9}
  },{
    approve:async()=>({success:false,uncertain:true,deferred:true,
      error:"No readback after lost ACK",stage:"storage"}),
    markApproved:async()=>{sideEffects++;throw Error("Must not update");},
    promote:async()=>{sideEffects++;throw Error("Must not promote");}
  });
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.proposalStateUpdated,false);
  assert.equal(sideEffects,0);
});
