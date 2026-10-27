import test from "node:test";
import assert from "node:assert/strict";
import {randomUUID} from "node:crypto";
import {runLearningWorker} from "../core/learning/learningWorker.js";
import {claimPendingLearningItems} from "../core/learning/learningQueueClaim.js";

function atomicQueue(ids){
  const records=new Map(ids.map(id=>[id,{
    id,action:"NEW_SKILL",status:"PENDING",
    event_json:{traceId:id,payload:{}},processed_at:null
  }]));
  let claims=0;
  const client={async rpc(name,args){
    assert.equal(name,"claim_pending_learning_queue");
    assert.ok(args.p_limit>0&&args.p_limit<=100);
    claims++;
    // Simulates one database statement: select and UPDATE are indivisible.
    const data=[];
    for(const entry of records.values()){
      if(entry.status!=="PENDING"||data.length>=args.p_limit)continue;
      entry.status="PROCESSING";
      entry.processed_at=new Date().toISOString();
      entry.claim_token=randomUUID();
      data.push(structuredClone(entry));
    }
    return {data,error:null};
  }};
  return {client,records,get claims(){return claims;}};
}

test("parallel Learning Workers cannot observe the same NEW_SKILL twice",async()=>{
  const store=atomicQueue(["queue-shared"]);
  let candidateWrites=0;
  const claim=()=>claimPendingLearningItems({client:store.client});
  const processItem=async item=>{
    assert.equal(item.status,"PROCESSING");
    assert.equal(item.event.traceId,item.id);
    candidateWrites++;
    await new Promise(resolve=>setTimeout(resolve,4));
    return {success:true,queueItemId:item.id};
  };
  const [first,second]=await Promise.all([
    runLearningWorker({claim,processItem}),
    runLearningWorker({claim,processItem})
  ]);
  assert.equal(first.processed+second.processed,1);
  assert.equal(first.successful+second.successful,1);
  assert.equal(candidateWrites,1,"Candidate Memory must be observed once");
  assert.equal(store.records.get("queue-shared").status,"PROCESSING");
  assert.equal(store.claims,2);
});

test("two simultaneous Workers partition a multi-item queue without duplication",async()=>{
  const store=atomicQueue(["a","b","c","d"]);
  const seen=[];
  const claim=()=>claimPendingLearningItems({client:store.client,limit:2});
  const processItem=async item=>{
    seen.push(item.id);
    return {success:true,queueItemId:item.id};
  };
  await Promise.all([
    runLearningWorker({claim,processItem}),
    runLearningWorker({claim,processItem})
  ]);
  assert.equal(seen.length,4);
  assert.deepEqual(new Set(seen),new Set(["a","b","c","d"]));
});

test("deferred claim returns to PENDING, and processing timestamp is released",async()=>{
  const store=atomicQueue(["deferred"]);
  const claim=()=>claimPendingLearningItems({client:store.client});
  const statusWrites=[];
  let calls=0;
  const processItem=async()=>{
    calls++;
    return calls===1?{success:true,deferred:true}:{success:true};
  };
  const updateQueue=async (id,status)=>{
    statusWrites.push([id,status]);
    const entry=store.records.get(id);
    entry.status=status;
    entry.processed_at=status==="PENDING"?null:new Date().toISOString();
    return {success:true};
  };
  const first=await runLearningWorker({claim,processItem,updateQueue});
  assert.equal(first.deferred,1);
  assert.equal(store.records.get("deferred").status,"PENDING");
  assert.equal(store.records.get("deferred").processed_at,null);
  const next=await runLearningWorker({claim,processItem,updateQueue});
  assert.equal(next.processed,1);
  assert.equal(calls,2);
  assert.deepEqual(statusWrites,[["deferred","PENDING"]]);
});

test("failed atomic claim fails closed without unprotected read or processing",async()=>{
  let processed=0;
  const client={async rpc(){return {data:null,error:{message:"RPC denied"}};}};
  const failure=await runLearningWorker({
    claim:()=>claimPendingLearningItems({client}),
    processItem:async()=>{processed++;return {success:true};}
  });
  assert.equal(failure.success,false);
  assert.match(failure.error,/RPC denied/);
  assert.equal(processed,0);
});

test("atomic claim API rejects missing result and bounds oversized limits",async()=>{
  const result=await claimPendingLearningItems({limit:1000,client:{
    async rpc(name,args){
      assert.equal(name,"claim_pending_learning_queue");
      assert.equal(args.p_limit,100);
      return {data:null,error:null};
    }
  }});
  assert.equal(result.success,false);
  assert.match(result.error,/Invalid atomic claim result/);
});
