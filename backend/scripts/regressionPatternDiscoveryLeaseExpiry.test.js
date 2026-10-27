import test from "node:test";
import assert from "node:assert/strict";
import {processQueueItem} from "../core/learning/learningWorker.js";
import {verifyLearningQueueClaim}
  from "../core/learning/learningQueueClaimCheck.js";
import {patternDiscoveryLeaseHarness}
  from "./fixtures/patternDiscoveryLeaseHarness.js";

function runDiscovery(h,{
  verifier=({queueItemId,claimToken})=>verifyLearningQueueClaim({
    queueItemId,claimToken,client:h.client
  })
}={}){
  const item={...h.item,claimToken:h.row.claim_token};
  const opts={
    replayGuard:async()=>({handled:false}),
    prepareDiscoveryStep:h.prepareDiscoveryStep,
    verifyClaim:verifier,
    prepareMemory:async()=>{h.counts.memory++;throw Error("Unexpected Candidate write");},
    proposalSaver:async()=>{h.counts.proposals++;throw Error("Unexpected Proposal");},
    proposalFailureHandler:async()=>{h.counts.retries++;throw Error("Unexpected retry");},
    updateStatus:async()=>{h.counts.updates++;throw Error("Unexpected Queue mutation");}
  };
  return processQueueItem(item,opts);
}
function assertNoMutation(h){
  assert.equal(h.counts.memory,0);
  assert.equal(h.counts.proposals,0);
  assert.equal(h.counts.retries,0);
  assert.equal(h.counts.updates,0);
}

test("Pattern Discovery expires while awaiting AI; delayed Worker cannot mutate",async()=>{
  const h=patternDiscoveryLeaseHarness(),work=runDiscovery(h);
  await h.started;
  h.expire(); // stale-recovery RPC invalidates the claim
  h.complete();
  const result=await work;
  assert.equal(result.success,false);
  assert.equal(result.stale,true);
  assert.match(result.reason,/expired/);
  assert.equal(h.counts.checks,1);
  assertNoMutation(h);
  assert.equal(h.row.status,"FAILED");
});

test("a rotated Queue token aborts an earlier Pattern result after AI",async()=>{
  const h=patternDiscoveryLeaseHarness(),work=runDiscovery(h);
  const original=h.row.claim_token;
  await h.started;
  h.rotate();
  assert.notEqual(original,h.row.claim_token);
  h.complete();
  const result=await work;
  assert.equal(result.success,false);
  assert.equal(result.stale,true);
  assertNoMutation(h);
  assert.equal(h.row.status,"PROCESSING");
});

test("no reusable Pattern may mark IGNORED only while claim is current",async()=>{
  const h=patternDiscoveryLeaseHarness(),claim=h.row.claim_token;
  const queueStatuses=[];
  const work=processQueueItem({...h.item,claimToken:claim},{
    replayGuard:async()=>({handled:false}),
    prepareDiscoveryStep:h.prepareDiscoveryStep,
    verifyClaim:p=>verifyLearningQueueClaim({...p,client:h.client}),
    updateStatus:async(id,status,token)=>{
      queueStatuses.push([id,status,token]);
      assert.equal(token,claim);
      h.row.status=status;h.row.claim_token=null;
      return {success:true};
    }
  });
  await h.started;
  h.complete({success:true,ignored:true,
    reason:"Extractor found no reusable result"});
  const result=await work;
  assert.equal(result.ignored,true);
  assert.equal(result.success,true);
  assert.deepEqual(queueStatuses,[[h.row.id,"IGNORED",claim]]);
  assert.equal(h.counts.checks,1);
});

test("DB ownership check failure during extraction fails closed",async()=>{
  const h=patternDiscoveryLeaseHarness();
  const work=runDiscovery(h,{verifier:async()=>({
    valid:false,stale:false,reason:"RPC permission denied"
  })});
  await h.started;
  h.complete();
  const result=await work;
  assert.equal(result.success,false);
  assert.equal(result.stale,false);
  assert.match(result.reason,/RPC permission denied/);
  assertNoMutation(h);
  assert.equal(h.row.status,"PROCESSING");
});

test("a failed extraction after lease expiry cannot retry or touch Queue",async()=>{
  const h=patternDiscoveryLeaseHarness(),work=runDiscovery(h);
  await h.started;
  h.expire();
  h.complete({success:false,retryLater:true,reason:"mocked HTTP 429"});
  const result=await work;
  assert.equal(result.success,false);
  assert.equal(result.stale,true);
  assert.equal(result.deferred,undefined);
  assertNoMutation(h);
});

test("invalid or missing claim token cannot pass read-only checkpoint",async()=>{
  const h=patternDiscoveryLeaseHarness();
  const noToken=await verifyLearningQueueClaim({
    queueItemId:h.row.id,claimToken:null,client:h.client
  });
  assert.equal(noToken.valid,false);
  assert.equal(h.counts.checks,0);
  const bad=await verifyLearningQueueClaim({
    queueItemId:h.row.id,claimToken:"00000000-0000-4000-8000-000000000001",
    client:h.client
  });
  assert.equal(bad.valid,false);
  const valid=await verifyLearningQueueClaim({
    queueItemId:h.row.id,claimToken:h.row.claim_token,client:h.client
  });
  assert.equal(valid.valid,true);
});
