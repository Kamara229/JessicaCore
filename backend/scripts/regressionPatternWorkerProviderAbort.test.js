import test from "node:test";
import assert from "node:assert/strict";
import {processQueueItem} from "../core/learning/learningWorker.js";
import {runPatternDiscoveryWithHeartbeat}
  from "../core/learning/patternDiscoveryHeartbeatRunner.js";
import {requestPatternExtractionAI} from "../ai/patternExtractionClient.js";

function runWithFakeProvider({heartbeat,timeoutMs=75}){
  const controllerState={requests:0,aborts:0,signal:null,
    candidate:0,proposal:0,queue:0,checks:0};
  let started;
  const active=new Promise(resolve=>{started=resolve;});
  const client={chat:{completions:{create(_body,opts){
    controllerState.requests++;
    controllerState.signal=opts.signal;
    started();
    return new Promise((_resolve,reject)=>{
      const rejectOnAbort=()=>{
        controllerState.aborts++;
        const error=new Error("SDK request cancelled");
        error.name="AbortError";
        reject(error);
      };
      opts.signal.addEventListener("abort",rejectOnAbort,{once:true});
      if(opts.signal.aborted)rejectOnAbort();
    });
  }}}};
  const item={id:"abort-worker",action:"PATTERN_DISCOVERY",
    claimToken:"00000000-0000-4000-8000-000000000045",
    event:{payload:{discovery:{}}}};
  const work=processQueueItem(item,{
    replayGuard:async()=>({handled:false}),
    prepareDiscoveryStep:async (_item,{signal})=>{
      await requestPatternExtractionAI({
        systemPrompt:"Verify evidence",evidenceText:"Test",
        signal,client
      });
    },
    superviseDiscovery:options=>runPatternDiscoveryWithHeartbeat({
      ...options,heartbeat,intervalMs:6,timeoutMs
    }),
    verifyClaim:async()=>{controllerState.checks++;return {valid:true};},
    prepareMemory:async()=>{controllerState.candidate++;throw Error("must not learn");},
    proposalSaver:async()=>{controllerState.proposal++;throw Error("must not save");},
    updateStatus:async()=>{controllerState.queue++;throw Error("must not update");}
  });
  return {work,active,state:controllerState};
}

for(const mode of ["owner lost","timeout"]){
  test("Worker aborts real SDK transport on "+mode+" without duplicate requests",async()=>{
    const flow=runWithFakeProvider({
      heartbeat:async()=>mode==="owner lost"
        ? {renewed:false,stale:true,reason:"claim rotated"}
        : {renewed:true},
      timeoutMs:mode==="owner lost"?80:30
    });
    await flow.active;
    const result=await flow.work;
    assert.equal(result.success,false);
    if(mode==="owner lost")assert.equal(result.stale,true);
    else assert.equal(result.timedOut,true);
    await new Promise(resolve=>setTimeout(resolve,0));
    assert.equal(flow.state.signal.aborted,true);
    assert.equal(flow.state.aborts,1);
    assert.equal(flow.state.requests,1);
    assert.equal(flow.state.candidate,0);
    assert.equal(flow.state.proposal,0);
    assert.equal(flow.state.queue,0);
    assert.equal(flow.state.checks,0);
  });
}
