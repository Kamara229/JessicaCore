import test from "node:test";
import assert from "node:assert/strict";
import {processQueueItem,runLearningWorker}
  from "../core/learning/learningWorker.js";
import {recordProposalSaveFailure}
  from "../core/learning/proposalSaveRetry.js";
import {proposalRetryStore} from "./fixtures/proposalRetryStore.js";

const skill={id:"disposable-retry-candidate",name:"Verified candidate",
  category:"research",confidence:0.85,occurrences:1,
  workflow:["check source"],examples:[{task:"verify",traceId:"retry-case"}]};
const item={id:"disposable-queue-retry",action:"NEW_SKILL",
  event:{traceId:"retry-case",action:"NEW_SKILL",confidence:0.85,
    payload:{skillCandidate:skill}}};

function setup({journal=true,limit=3}={}){
  const store=proposalRetryStore(item.id,{journal,limit});
  let memoryWrites=0,saverCalls=0,updateCalls=0;
  const options={
    replayGuard:async()=>({handled:false}),
    prepareMemory:async queue=>{
      // An already journaled Candidate is loaded without another write.
      if(memoryWrites===0)memoryWrites++;
      return {success:true,queueItem:queue,observed:false,
        candidate:skill,created:false,merged:false};
    },
    proposalSaver:async proposal=>{
      saverCalls++;
      if(saverCalls<=2)return {success:false,error:"mocked Supabase write failure"};
      store.proposals.set(item.id,proposal);
      return {success:true,proposal};
    },
    proposalFailureHandler:params=>recordProposalSaveFailure({
      ...params,client:store.client
    }),
    updateStatus:async(id,status)=>{
      updateCalls++;
      assert.equal(id,item.id);
      store.queue.status=status;
      return {success:true};
    }
  };
  return {store,options,get memoryWrites(){return memoryWrites;},
    get saverCalls(){return saverCalls;},get updateCalls(){return updateCalls;}};
}

test("Proposal fails twice after Candidate journal; third attempt saves once",async()=>{
  const mock=setup();
  const first=await processQueueItem(item,mock.options);
  assert.equal(first.success,true);
  assert.equal(first.deferred,true);
  assert.equal(first.retryScheduled,true);
  assert.equal(first.queueAlreadyReleased,true);
  assert.equal(first.attempts,1);
  assert.equal(mock.store.queue.status,"PENDING");
  mock.store.reclaim();
  const second=await processQueueItem(item,mock.options);
  assert.equal(second.success,true);
  assert.equal(second.deferred,true);
  assert.equal(second.attempts,2);
  assert.equal(mock.store.queue.status,"PENDING");
  mock.store.reclaim();
  const final=await processQueueItem(item,mock.options);
  assert.equal(final.success,true);
  assert.equal(final.proposalId,mock.store.proposals.get(item.id).id);
  assert.equal(mock.store.queue.status,"PROPOSED");
  assert.equal(mock.store.attempts,2);
  assert.equal(mock.store.proposals.size,1);
  assert.equal(mock.memoryWrites,1);
  assert.equal(mock.updateCalls,1);
});

test("repeated failures stop at three attempts with no duplicate candidate",async()=>{
  const mock=setup();
  mock.options.proposalSaver=async()=>({
    success:false,error:"Supabase unavailable"
  });
  const results=[];
  for(let i=0;i<3;i++){
    if(i>0)mock.store.reclaim();
    results.push(await processQueueItem(item,mock.options));
  }
  assert.deepEqual(results.map(r=>r.attempts),[1,2,3]);
  assert.equal(results[2].exhausted,true);
  assert.equal(results[2].success,false);
  assert.equal(mock.store.queue.status,"FAILED");
  assert.equal(mock.store.attempts,3);
  assert.equal(mock.memoryWrites,1);
  assert.equal(mock.store.proposals.size,0);
});

test("lost Proposal write acknowledgement recognizes existing Proposal",async()=>{
  const mock=setup();
  mock.store.proposals.set(item.id,{id:"existing-saved-proposal"});
  mock.options.proposalSaver=async()=>{throw new Error("response lost");};
  const result=await processQueueItem(item,mock.options);
  assert.equal(result.success,true);
  assert.equal(result.existing,true);
  assert.equal(result.proposalId,"existing-saved-proposal");
  assert.equal(result.recovered,true);
  assert.equal(mock.store.queue.status,"PROPOSED");
  assert.equal(mock.store.attempts,0);
  assert.equal(mock.store.proposals.size,1);
});

test("missing journal proof cannot release or retry a claimed queue item",async()=>{
  const mock=setup({journal:false});
  const result=await processQueueItem(item,mock.options);
  assert.equal(result.success,false);
  assert.equal(result.recoveryBlocked,true);
  assert.equal(mock.store.queue.status,"PROCESSING");
  assert.equal(mock.store.attempts,0);
});

test("RPC error fails closed; Worker must not overwrite Queue state",async()=>{
  const mock=setup();
  mock.options.proposalFailureHandler=async params=>
    recordProposalSaveFailure({...params,client:{
      async rpc(){return {data:null,error:{message:"RPC unavailable"}};}
    }});
  const result=await processQueueItem(item,mock.options);
  assert.equal(result.success,false);
  assert.equal(result.recoveryBlocked,true);
  assert.equal(mock.store.queue.status,"PROCESSING");
  assert.equal(mock.store.proposals.size,0);
});

test("Worker does not release an already atomically released queue item twice",async()=>{
  const updates=[];
  const result=await runLearningWorker({
    claim:async()=>({success:true,items:[item]}),
    processItem:async()=>({success:true,deferred:true,
      retryScheduled:true,queueAlreadyReleased:true}),
    updateQueue:async(id,status)=>{updates.push([id,status]);return {success:true};}
  });
  assert.equal(result.success,true);
  assert.equal(result.deferred,1);
  assert.deepEqual(updates,[]);
});

test("bounded adapter rejects an invalid or unknown RPC response",async()=>{
  const bad=await recordProposalSaveFailure({queueItemId:item.id,client:{
    async rpc(){return {data:{status:"retry",attempts:99},error:null};}
  }});
  assert.equal(bad.success,false);
  const missing=await recordProposalSaveFailure({queueItemId:"",client:{
    async rpc(){throw Error("must not call");}
  }});
  assert.equal(missing.success,false);
});
