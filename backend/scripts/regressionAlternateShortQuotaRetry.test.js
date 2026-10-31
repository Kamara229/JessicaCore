import test from "node:test";
import assert from "node:assert/strict";
import {isShortAlternateRateLimit,requestAlternateWithShortRetry} from "../ai/alternateShortQuotaRetry.js";
import {withDailyQuotaModelFallback} from "../ai/dailyQuotaModelFallback.js";

const tpm=(retry="2")=>Object.assign(new Error(
 "429 Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM). Please try again in 1.52s."
),{status:429,headers:{"retry-after":retry}});
const daily=()=>Object.assign(new Error("429 tokens per day (TPD) exhausted"),{status:429});

test("120b TPM failure is an immediate one-retry case, no extra model escalation",async()=>{
 let attempts=0,delays=[];
 const result=await requestAlternateWithShortRetry(async model=>{
   attempts++;assert.equal(model,"openai/gpt-oss-120b");
   if(attempts===1)throw tpm();
   return "model-ready";
 },"openai/gpt-oss-120b",{sleep:async ms=>delays.push(ms)});
 assert.equal(result,"model-ready");
 assert.equal(attempts,2);
 assert.deepEqual(delays,[2350]);
});

test("daily quotas, unsafe shapes and long cooldown never retry independent model",async()=>{
 const cases=[
  daily(),
  Object.assign(new Error("Request too large: expected output tokens exceed ceiling"),{status:429,headers:{"retry-after":"1"}}),
  tpm("45"),
  Object.assign(new Error("Something went wrong"),{status:500})
 ];
 for(const err of cases){
   let called=0,waited=0;
   assert.equal(isShortAlternateRateLimit(err),false);
   await assert.rejects(()=>requestAlternateWithShortRetry(async()=>{
     called++;throw err;
   },"openai/gpt-oss-120b",{sleep:async()=>{waited++;}}),e=>e===err);
   assert.equal(called,1);assert.equal(waited,0);
 }
});

test("one retry only: second short TPM failure propagates",async()=>{
 let called=0,waited=0;
 await assert.rejects(()=>requestAlternateWithShortRetry(async()=>{
   called++;throw tpm("1");
 },"openai/gpt-oss-120b",{sleep:async()=>{waited++;}}),e=>e.status===429);
 assert.equal(called,2);assert.equal(waited,1);
});

test("end-to-end 20b daily quota to 120b short TPM recovery",async()=>{
 let original=0,alternate=0;
 const result=await withDailyQuotaModelFallback({
   primaryModel:"openai/gpt-oss-20b",
   primaryRequest:async()=>{original++;throw daily();},
   alternateRequest:async()=>{alternate++;
      if(alternate===1)throw tpm("0");
      return {text:"correct answer after temporary quota reset"};
   }
 });
 assert.equal(result.text,"correct answer after temporary quota reset");
 assert.equal(original,1);assert.equal(alternate,2);
});
