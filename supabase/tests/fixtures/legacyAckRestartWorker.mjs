import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {legacyClaimReceiptLookup} from "./legacyClaimReceiptPg.mjs";
import {assertIsolatedCi} from "./candidateRacePgClient.mjs";

assertIsolatedCi();
const mode=process.argv[2];
assert.ok(["offline","corrupt","restored"].includes(mode));
const owner="ci-legacy-ack-owner",skill="ci-legacy-ack-skill";
const receipt=legacyClaimReceiptLookup(owner);
receipt.setOutage(mode==="offline");
const lookup=async id=>{
  const found=await receipt.lookup(id);
  if(mode==="corrupt" && found)
    return {...found,payload:{...found.payload,id:"wrong-skill"}};
  return found;
};
let writes=0;
const result=await persistExperienceSkill({
  id:skill,name:"Legacy owner claim migration with lost ACK",
  version:2,previousVersion:1,mode:"update",enabled:false,
  workflow:["Confirm unique durable Skill claim on restart"],
  metadata:{proposalId:owner}
},{
  lookup,
  save:async()=>{writes++;throw Error("unexpected repeat Skill write");}
});
assert.equal(writes,0,"restarted worker must never write another version");
assert.equal(receipt.calls,1);
if(mode==="restored"){
  assert.equal(result.success,true);
  assert.equal(result.existing,true);
  assert.equal(result.skillId,skill);
  assert.equal(result.version,2);
  assert.equal(result.storage,"idempotency-precheck");
}else{
  assert.equal(result.success,false);
  assert.equal(result.uncertain,true);
  assert.equal(result.existing,false);
}
console.log("PASS legacy receipt restart "+mode);
