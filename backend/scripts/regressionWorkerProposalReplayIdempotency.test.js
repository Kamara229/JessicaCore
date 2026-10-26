import test from "node:test";
import assert from "node:assert/strict";
import {checkLearningWorkerReplay}
  from "../core/learning/workerReplayGuard.js";
import {processQueueItem}
  from "../core/learning/learningWorker.js";

const makeQueue=(id)=>({
  id,action:"SKILL_IMPROVEMENT",status:"PENDING",
  event:{traceId:"shared-trace",skillId:id,
    action:"SKILL_IMPROVEMENT",payload:{}}
});

test("Worker replay skips an already APPROVED Proposal without re-learning",async()=>{
  const queue=makeQueue("queue-skill-v1");
  const approved={id:"saved-proposal-v1",queueItemId:queue.id,
    status:"APPROVED",action:"SKILL_IMPROVEMENT"};
  const calls=[];
  const fakeDeps={
    findProposal:async id=>{calls.push(["find",id]);
      return {success:true,found:true,proposal:structuredClone(approved)};},
    updateQueue:async (id,status)=>{calls.push(["queue",id,status]);
      return {success:true};}
  };
  const run=()=>processQueueItem(queue,{replayGuard:
    item=>checkLearningWorkerReplay(item,fakeDeps)});
  const first=await run(),second=await run();
  for(const result of [first,second]){
    assert.equal(result.success,true);
    assert.equal(result.existing,true);
    assert.equal(result.proposalId,approved.id);
    assert.equal(result.proposal.status,"APPROVED",
      "Never reopen a previously approved proposal");
    assert.equal(result.queueUpdated,true);
  }
  assert.deepEqual(calls,[
    ["find",queue.id],["queue",queue.id,"PROPOSED"],
    ["find",queue.id],["queue",queue.id,"PROPOSED"]
  ]);
  // No injected proposal-creation, storage, candidate or approval tools are
  // used: replay guard short-circuits before the normal Worker route.
});

test("two independent Skills in one Trace replay their existing IDs only",async()=>{
  const ids=["queue-skill-v1","queue-skill-v2"];
  const stored=new Map(ids.map((id,index)=>[
    id,{id:"saved-proposal-"+index,queueItemId:id,status:"PROCESSING"}
  ]));
  const changes=[];
  const deps={
    findProposal:async id=>({success:true,found:true,
      proposal:structuredClone(stored.get(id))}),
    updateQueue:async (id,status)=>{
      changes.push([id,status]);return {success:true};}
  };
  const results=await Promise.all(ids.map(id=>
    processQueueItem(makeQueue(id),{replayGuard:
      item=>checkLearningWorkerReplay(item,deps)})));
  assert.deepEqual(results.map(r=>r.proposalId),
    ["saved-proposal-0","saved-proposal-1"]);
  assert.ok(results.every(r=>r.existing===true));
  assert.ok(results.every(r=>r.proposal.status==="PROCESSING"));
  assert.deepEqual(changes,ids.map(id=>[id,"PROPOSED"]));
});

test("proposal lookup failure defers Worker, without creating or rejecting Skill",async()=>{
  const item=makeQueue("queue-lost-network");
  const result=await processQueueItem(item,{replayGuard:
    x=>checkLearningWorkerReplay(x,{
      findProposal:async()=>({success:false,error:"mocked network failure"}),
      updateQueue:async()=>{throw new Error("Must not mutate queue");}
    })});
  assert.equal(result.success,true);
  assert.equal(result.deferred,true);
  assert.equal(result.proposalId,undefined);
});

test("lookup no-match preserves normal first-processing path",async()=>{
  const result=await checkLearningWorkerReplay(makeQueue("new-event"),{
    findProposal:async()=>({success:true,found:false,proposal:null}),
    updateQueue:async()=>{throw new Error("must not update unmatched queue");}
  });
  assert.deepEqual(result,{handled:false});
});

test("queue update failure keeps recovered proposal immutable for retry",async()=>{
  const item=makeQueue("queue-update-fail");
  const result=await processQueueItem(item,{replayGuard:
    x=>checkLearningWorkerReplay(x,{
      findProposal:async()=>({success:true,found:true,proposal:{
        id:"already-approved",queueItemId:item.id,status:"APPROVED"}}),
      updateQueue:async()=>({success:false,error:"write failed"})
    })});
  assert.equal(result.success,false);
  assert.equal(result.existing,true);
  assert.equal(result.proposalId,"already-approved");
  assert.equal(result.proposal.status,"APPROVED");
  assert.equal(result.queueUpdated,false);
});
