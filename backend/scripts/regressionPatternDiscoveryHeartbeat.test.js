import test from "node:test";
import assert from "node:assert/strict";
import {renewLearningQueueClaim}
  from "../core/learning/learningQueueHeartbeat.js";
import {runPatternDiscoveryWithHeartbeat}
  from "../core/learning/patternDiscoveryHeartbeatRunner.js";
import {processQueueItem} from "../core/learning/learningWorker.js";
import {patternHeartbeatLeaseClient}
  from "./fixtures/patternHeartbeatLeaseClient.js";

function supervisor(store,{intervalMs=7,timeoutMs=95}={}){
  return options=>runPatternDiscoveryWithHeartbeat({
    ...options,intervalMs,timeoutMs,
    heartbeat:p=>renewLearningQueueClaim({...p,client:store.client})
  });
}
function blockedWorker(store){
  const writes={candidate:0,proposal:0,queue:0,check:0};
  let started;
  const startedPromise=new Promise(resolve=>{started=resolve;});
  const item={...store.item,event:{payload:{discovery:{}}}};
  const result=processQueueItem(item,{
    replayGuard:async()=>({handled:false}),
    prepareDiscoveryStep:async()=>{
      started();
      return new Promise(()=>{}); // Simulate stalled external AI provider.
    },
    superviseDiscovery:supervisor(store),
    verifyClaim:async()=>{writes.check++;return {valid:true};},
    prepareMemory:async()=>{writes.candidate++;throw Error("Candidate mutation");},
    proposalSaver:async()=>{writes.proposal++;throw Error("Proposal mutation");},
    updateStatus:async()=>{writes.queue++;throw Error("Queue mutation");}
  });
  return {result,startedPromise,writes};
}

test("periodic heartbeats keep an owned long-running discovery claim alive",async()=>{
  const store=patternHeartbeatLeaseClient();
  let beats=0,finish;
  const work=new Promise(resolve=>{finish=resolve;});
  const result=await runPatternDiscoveryWithHeartbeat({
    item:store.item,
    heartbeat:async p=>{
      const r=await renewLearningQueueClaim({...p,client:store.client});
      beats++;
      if(beats===3)finish({success:true});
      return r;
    },
    work:()=>work,intervalMs:6,timeoutMs:95
  });
  assert.equal(result.ok,true);
  assert.equal(result.value.success,true);
  assert.ok(beats>=3);
  assert.equal(store.calls,beats);
  assert.equal(store.row.status,"PROCESSING");
  assert.equal(store.row.claim_token!==null,true);
});

test("token rotation during in-flight AI stops Worker without writes",async()=>{
  const store=patternHeartbeatLeaseClient();
  const runner=blockedWorker(store);
  await runner.startedPromise;
  const previous=store.row.claim_token;
  store.rotate();
  assert.notEqual(store.row.claim_token,previous);
  const result=await runner.result;
  assert.equal(result.success,false);
  assert.equal(result.stale,true);
  assert.match(result.reason,/claim_expired_or_stale/);
  assert.deepEqual(runner.writes,{candidate:0,proposal:0,queue:0,check:1});
  assert.ok(store.calls>=1);
});

test("expired lease stops Worker even if external extraction never completes",async()=>{
  const store=patternHeartbeatLeaseClient();
  const runner=blockedWorker(store);
  await runner.startedPromise;
  store.expire();
  const result=await runner.result;
  assert.equal(result.success,false);
  assert.equal(result.stale,true);
  assert.deepEqual(runner.writes,{candidate:0,proposal:0,queue:0,check:1});
});

test("heartbeat RPC error prevents unverified continued learning",async()=>{
  const store=patternHeartbeatLeaseClient();
  const result=await runPatternDiscoveryWithHeartbeat({
    item:store.item,intervalMs:5,timeoutMs:75,
    heartbeat:async()=>({renewed:false,reason:"network unavailable"}),
    work:async()=>new Promise(()=>{})
  });
  assert.equal(result.ok,false);
  assert.equal(result.stale,false);
  assert.match(result.reason,/network unavailable/);
});

test("hard discovery timeout terminates supervision and cleans timers",async()=>{
  const store=patternHeartbeatLeaseClient();
  const before=store.calls;
  const result=await runPatternDiscoveryWithHeartbeat({
    item:store.item,intervalMs:6,timeoutMs:32,
    heartbeat:p=>renewLearningQueueClaim({...p,client:store.client}),
    work:async()=>new Promise(()=>{})
  });
  assert.equal(result.ok,false);
  assert.equal(result.timedOut,true);
  assert.match(result.reason,/timed out/);
  assert.ok(store.calls>before);
  const stopped=store.calls;
  await new Promise(resolve=>setTimeout(resolve,24));
  assert.equal(store.calls,stopped,"Heartbeat timer must stop after timeout");
});

test("missing token must not run or start any background timers",async()=>{
  let invoked=false;
  const result=await runPatternDiscoveryWithHeartbeat({
    item:{id:"invalid"},
    work:()=>{invoked=true;return {success:true};},
    heartbeat:async()=>{throw Error("Unexpected heartbeat");},
    intervalMs:5,timeoutMs:20
  });
  assert.equal(result.ok,false);
  assert.equal(invoked,false);
});
