import test from "node:test";
import assert from "node:assert/strict";
import {runLearningWorker} from "../core/learning/learningWorker.js";
import {LEARNING_WORKER_CLAIM_LIMIT}
  from "../core/learning/learningWorkerClaimBudget.js";

test("sequential Worker never preclaims a 100-item backlog",async()=>{
  const pending=Array.from({length:19},(_,i)=>({
    id:"queue-"+i,action:"SKILL_IMPROVEMENT"
  }));
  const processed=[],claimed=[];
  const claim=async ({limit}={})=>{
    assert.equal(limit,LEARNING_WORKER_CLAIM_LIMIT);
    const items=pending.splice(0,limit);
    claimed.push(items.length);
    return {success:true,items};
  };
  const processItem=async item=>{
    processed.push(item.id);
    return {success:true,queueItemId:item.id};
  };
  const one=await runLearningWorker({claim,processItem});
  assert.equal(one.success,true);
  assert.equal(one.processed,8);
  assert.equal(pending.length,11);
  const two=await runLearningWorker({claim,processItem});
  const three=await runLearningWorker({claim,processItem});
  assert.deepEqual([one.processed,two.processed,three.processed],[8,8,3]);
  assert.deepEqual(claimed,[8,8,3]);
  assert.equal(new Set(processed).size,19);
  assert.equal(pending.length,0);
});

test("bounded claims fit the current recovery lease even at extraction timeout",()=>{
  const longestExtractionMinutes=5;
  const minimumRecoveryLeaseMinutes=120;
  assert.ok(LEARNING_WORKER_CLAIM_LIMIT*longestExtractionMinutes
    <minimumRecoveryLeaseMinutes/2);
});
