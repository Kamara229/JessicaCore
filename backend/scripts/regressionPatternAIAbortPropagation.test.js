import test from "node:test";
import assert from "node:assert/strict";
import {executeAIWithRetry} from "../ai/aiRetry.js";
import {requestPatternExtractionAI} from "../ai/patternExtractionClient.js";
import {requestPatternExtraction}
  from "../core/learning/patternExtraction/patternRequest.js";
import {extractExperiencePattern}
  from "../core/learning/experiencePatternExtractor.js";
import {processPatternDiscovery}
  from "../core/learning/patternDiscoveryWorker.js";

test("AI transport receives signal in SDK options; abort stops first request and retries",async()=>{
  const controller=new AbortController();
  let calls=0,started;
  const wasStarted=new Promise(resolve=>{started=resolve;});
  const client={chat:{completions:{create(_body,options){
    calls++;
    assert.equal(options.signal,controller.signal);
    assert.equal(options.maxRetries,0);
    started();
    return new Promise((_resolve,reject)=>{
      const fail=()=>{const error=new Error("aborted by provider SDK");
        error.name="AbortError";reject(error);};
      options.signal.addEventListener("abort",fail,{once:true});
      if(options.signal.aborted)fail();
    });
  }}}};
  const work=requestPatternExtractionAI({
    systemPrompt:"Test system",evidenceText:"Test evidence",
    signal:controller.signal,client
  });
  await wasStarted;
  controller.abort();
  await assert.rejects(work,/abort/i);
  assert.equal(calls,1,"abort must stop SDK and technical retry");
});

test("abort before transport starts sends zero SDK requests",async()=>{
  const controller=new AbortController();
  controller.abort();
  let calls=0;
  await assert.rejects(requestPatternExtractionAI({
    systemPrompt:"Test system",evidenceText:"Test evidence",
    signal:controller.signal,
    client:{chat:{completions:{create(){calls++;}}}}
  }),/abort/i);
  assert.equal(calls,0);
});

test("abort during technical 503 backoff prevents next provider attempt",async()=>{
  const controller=new AbortController();
  let calls=0,failed;
  const firstFailure=new Promise(resolve=>{failed=resolve;});
  const work=executeAIWithRetry(async()=>{
    calls++;
    const error=new Error("Service unavailable");
    error.status=503;
    failed();
    throw error;
  },{label:"mock Pattern AI",maxAttempts:3,signal:controller.signal});
  await firstFailure;
  controller.abort();
  await assert.rejects(work,e=>e.name==="AbortError");
  assert.equal(calls,1);
});

test("Pattern request forwards signal and suppresses a response received after cancellation",async()=>{
  const controller=new AbortController();
  let seen=null;
  const response=await requestPatternExtraction(
    {task:"Verify source"},
    {signal:controller.signal,isAvailable:()=>true,
      requestAI:async ({signal})=>{
        seen=signal;
        controller.abort();
        return {choices:[{message:{content:'{"reusable":true}'}}]};
      }}
  );
  assert.equal(seen,controller.signal);
  assert.equal(response.success,false);
  assert.equal(response.cancelled,true);
  assert.equal(response.retryLater,false);
});

test("Extractor never parses or promotes aborted response",async()=>{
  const controller=new AbortController();
  const result=await extractExperiencePattern({
    evidence:{task:"Verify source"},
    signal:controller.signal,
    request:async (_evidence,{signal})=>{
      assert.equal(signal,controller.signal);
      controller.abort();
      return {success:true,rawText:'{"reusable":true}'};
    }
  });
  assert.equal(result.success,false);
  assert.equal(result.stage,"cancelled");
  assert.equal(result.pattern,null);
});

test("Pattern Discovery forwards signal and cannot build a candidate after cancellation",async()=>{
  const controller=new AbortController();
  const item={
    id:"mock-discovery",action:"PATTERN_DISCOVERY",
    event:{payload:{discovery:{evidence:{
      task:"Read a source",answer:"Checked source",
      result:{success:true,verified:true,status:"COMPLETED"},
      validation:{valid:true,degraded:false}
    }}}}
  };
  let calls=0;
  const result=await processPatternDiscovery(item,{
    signal:controller.signal,
    extractor:async ({signal})=>{
      assert.equal(signal,controller.signal);
      calls++;
      controller.abort();
      return {success:true,reusable:true,pattern:{name:"should never become candidate"}};
    }
  });
  assert.equal(calls,1);
  assert.equal(result.success,false);
  assert.equal(result.cancelled,true);
  assert.equal(result.queueItem,undefined);
});

test("uncancelled request retains normal JSON response handling",async()=>{
  const controller=new AbortController();
  const response=await requestPatternExtraction({task:"Check text"},{
    signal:controller.signal,isAvailable:()=>true,
    requestAI:async ({signal})=>{
      assert.equal(signal,controller.signal);
      return {choices:[{message:{content:'{"reusable":false}'}}]};
    }
  });
  assert.equal(response.success,true);
  assert.equal(response.rawText,'{"reusable":false}');
});
