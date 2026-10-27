import test from "node:test";
import assert from "node:assert/strict";
import {processQueueItem} from "../core/learning/learningWorker.js";

const token="00000000-0000-4000-8000-000000000046";
function runPreflight(verify){
  const calls={checks:0,ai:0,memory:0,proposal:0,queue:0};
  const item={id:"preflight-check",claimToken:token,
    action:"PATTERN_DISCOVERY",event:{payload:{discovery:{}}}};
  const result=processQueueItem(item,{
    replayGuard:async()=>({handled:false}),
    verifyClaim:async args=>{
      calls.checks++;
      assert.deepEqual(args,{queueItemId:item.id,claimToken:token});
      return verify();
    },
    prepareDiscoveryStep:async()=>{
      calls.ai++;
      return {success:true,ignored:true,reason:"No reusable pattern"};
    },
    prepareMemory:async()=>{calls.memory++;throw Error("Unwanted candidate");},
    proposalSaver:async()=>{calls.proposal++;throw Error("Unwanted proposal");},
    updateStatus:async()=>{
      calls.queue++;
      return {success:true};
    }
  });
  return {result,calls};
}

for(const [name,verify,stale] of [
  ["rotated token",()=>({valid:false,stale:true,reason:"Claim rotated"}),true],
  ["database RPC outage",()=>({valid:false,stale:false,reason:"RPC unavailable"}),false],
  ["thrown verification error",()=>{throw Error("Connection lost");},false]
]){
  test("stale preflight prevents AI and all writes: "+name,async()=>{
    const flow=runPreflight(verify);
    const result=await flow.result;
    assert.equal(result.success,false);
    assert.equal(result.stale,stale);
    assert.equal(flow.calls.checks,1);
    assert.equal(flow.calls.ai,0);
    assert.equal(flow.calls.memory,0);
    assert.equal(flow.calls.proposal,0);
    assert.equal(flow.calls.queue,0);
  });
}

test("valid preflight still requires a second ownership check after AI",async()=>{
  const flow=runPreflight(()=>({valid:true,stale:false}));
  const result=await flow.result;
  assert.equal(result.success,true);
  assert.equal(result.ignored,true);
  assert.deepEqual(flow.calls,{
    checks:2,ai:1,memory:0,proposal:0,queue:1
  });
});
