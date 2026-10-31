import test from "node:test";
import assert from "node:assert/strict";
import {withDailyQuotaModelFallback,QUOTA_FALLBACK_MODEL} from "../ai/dailyQuotaModelFallback.js";
import {getPlannerModel,getPlannerMaxCompletionTokens} from "../ai/plannerClient.js";
import {getComposerModel} from "../ai/composerClient.js";

test("primary model works unchanged while available",async()=>{
 let alternate=0;
 const result=await withDailyQuotaModelFallback({
   primaryModel:getPlannerModel(),primaryRequest:async()=>({model:"primary"}),
   alternateRequest:async()=>{alternate++;return {model:"alternate"}}
 });
 assert.equal(result.model,"primary");assert.equal(alternate,0);
});

test("Groq 20B TPD failure uses one independent model, without AI retrying the exhausted model",async()=>{
 let primary=0,alternate=0;
 const result=await withDailyQuotaModelFallback({
   primaryModel:"openai/gpt-oss-20b",
   primaryRequest:async()=>{primary++;throw Object.assign(new Error("429 Rate limit on tokens per day (TPD)"),{status:429})},
   alternateRequest:async(model)=>{alternate++;assert.equal(model,"openai/gpt-oss-120b");return {model}}
 });
 assert.equal(result.model,QUOTA_FALLBACK_MODEL);
 assert.equal(primary,1);assert.equal(alternate,1);
});

test("other quotas and technical/semantic failures never switch model",async()=>{
 const cases=[
  Object.assign(new Error("429 tokens per minute (TPM)"),{status:429}),
  Object.assign(new Error("400 invalid JSON"),{status:400}),
  Object.assign(new Error("Server error"),{status:500})
 ];
 for(const error of cases){
  let calls=0;
  await assert.rejects(()=>withDailyQuotaModelFallback({
    primaryModel:"openai/gpt-oss-20b",
    primaryRequest:async()=>{throw error},
    alternateRequest:async()=>{calls++;return "bad"}
  }),err=>err===error);
  assert.equal(calls,0);
 }
});

test("a model already on 120B cannot escalate or repeat on its daily quota",async()=>{
 const error=Object.assign(new Error("429 tokens per day TPD"),{status:429});
 let calls=0;
 await assert.rejects(()=>withDailyQuotaModelFallback({
   primaryModel:"openai/gpt-oss-120b",
   primaryRequest:async()=>{throw error},
   alternateRequest:async()=>{calls++;return "bad"}
 }),err=>err===error);
 assert.equal(calls,0);
});

test("planner and composer keep their existing primary model identity and budget",()=>{
 assert.equal(getPlannerModel(),"openai/gpt-oss-20b");
 assert.equal(getPlannerMaxCompletionTokens(),800);
 assert.equal(getComposerModel(),"openai/gpt-oss-20b");
});
