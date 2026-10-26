import test from "node:test";
import assert from "node:assert/strict";
import {getAIProviderAvailability,registerAIProviderLimit,
    throwIfAIProviderCoolingDown}
 from "../ai/providerAvailability.js";
import {executeAIWithRetry}
 from "../ai/retry/retryExecutor.js";
import {getRetryAfterDetails}
 from "../ai/retry/retryAfter.js";
import {isRetryableAIError}
 from "../ai/retry/retryClassifier.js";

const BASE=Date.now()+3_600_000;
function clock(t,index){
    let now=BASE+index*3_600_000;
    t.mock.method(Date,"now",()=>now);
    t.mock.method(console,"warn",()=>{});
    return {advance:delta=>{now+=delta;},now:()=>now};
}
function quota(message,header){
    const error=new Error(message);
    error.status=429;
    if(header!==undefined)error.headers={"retry-after":header};
    return error;
}

test("short minute-limit Retry-After stays local and never opens global circuit",t=>{
    clock(t,1);
    const error=quota("tokens per minute exceeded","2");
    assert.deepEqual(getRetryAfterDetails(error),{
        milliseconds:2000,source:"retry-after"});
    assert.equal(isRetryableAIError(error),true);
    assert.equal(registerAIProviderLimit(error).blocked,false);
    assert.equal(getAIProviderAvailability().blocked,false);
});

test("daily quota opens shared breaker; repeated sessions issue zero extra calls",async t=>{
    clock(t,2);
    const error=quota("tokens per day (TPD) exceeded");
    let attempts=0;
    await assert.rejects(executeAIWithRetry(async()=>{
        attempts++;
        throw error;
    },{maxAttempts:5,label:"first session"}),e=>e===error);
    assert.equal(attempts,1);
    const current=getAIProviderAvailability();
    assert.equal(current.blocked,true);
    assert.equal(current.reason,"daily-quota");
    assert.ok(current.remainingMs>=59000&&current.remainingMs<=60000);
    await assert.rejects(executeAIWithRetry(async()=>{
        attempts++;return "wrong";
    },{maxAttempts:5,label:"second session"}),err=>
        err.code==="AI_PROVIDER_COOLDOWN"&&err.providerCooldown===true);
    assert.equal(attempts,1);
});

test("long Retry-After prevents retries until expiry, then resumes",async t=>{
    const time=clock(t,3);
    const error=quota("requests per minute exceeded","120");
    assert.deepEqual(getRetryAfterDetails(error),{
        milliseconds:120000,source:"retry-after"});
    let attempts=0;
    await assert.rejects(executeAIWithRetry(async()=>{
        attempts++;throw error;
    },{maxAttempts:6}),e=>e===error);
    assert.equal(attempts,1);
    assert.equal(getAIProviderAvailability().reason,"long-rate-limit");
    assert.equal(getAIProviderAvailability().remainingMs,120000);
    time.advance(119999);
    assert.equal(getAIProviderAvailability().blocked,true);
    await assert.rejects(executeAIWithRetry(async()=>{
        attempts++;return "not allowed";
    }),e=>e.code==="AI_PROVIDER_COOLDOWN");
    assert.equal(attempts,1);
    time.advance(1);
    assert.equal(getAIProviderAvailability().blocked,false);
    const resumed=await executeAIWithRetry(async()=>{
        attempts++;return "resumed";
    });
    assert.equal(resumed,"resumed");
    assert.equal(attempts,2);
});

test("Retry-After bounded to 15 minutes and repeated short limit cannot shorten",t=>{
    const time=clock(t,4);
    const long=quota("tokens per minute exceeded","2000");
    const status=registerAIProviderLimit(long);
    assert.equal(status.remainingMs,15*60_000);
    assert.equal(status.source,"retry-after");
    time.advance(1000);
    const shorter=quota("requests per minute exceeded","120");
    const next=registerAIProviderLimit(shorter);
    assert.equal(next.remainingMs,15*60_000-1000);
    time.advance(15*60_000-1001);
    assert.equal(getAIProviderAvailability().blocked,true);
    time.advance(1);
    assert.equal(getAIProviderAvailability().available,true);
});

test("concurrent request opens breaker while first waits: no second network call",async t=>{
    clock(t,5);
    const minute=quota("tokens per minute exceeded","0");
    const daily=quota("requests per day quota exceeded");
    let operations=0;
    let switched=false;
    await assert.rejects(executeAIWithRetry(async()=>{
        operations++;
        setTimeout(()=>{
            switched=true;
            registerAIProviderLimit(daily);
        },10);
        throw minute;
    },{maxAttempts:3,label:"awaiting rate limit"}),e=>
        e.status===429&&e.code==="AI_PROVIDER_COOLDOWN");
    assert.equal(switched,true);
    assert.equal(operations,1);
    assert.equal(getAIProviderAvailability().reason,"daily-quota");
});

test("hard output-size 429 is never retried and does not open cooldown",async t=>{
    clock(t,6);
    const error=quota("request too large; reduce max_tokens","120");
    let operations=0;
    await assert.rejects(executeAIWithRetry(async()=>{
        operations++;throw error;
    },{maxAttempts:7}),e=>e===error);
    assert.equal(operations,1);
    assert.equal(isRetryableAIError(error),false);
    assert.equal(getAIProviderAvailability().blocked,false);
});

test("provider cooldown rejection exposes remaining delay without changing proof",t=>{
    const time=clock(t,7);
    const status=registerAIProviderLimit(quota("tokens per day", "90"));
    assert.equal(status.blocked,true);
    time.advance(1000);
    assert.throws(()=>throwIfAIProviderCoolingDown("signed doc"),err=>
        err.code==="AI_PROVIDER_COOLDOWN"&&
        err.status===429&&err.providerCooldown===true&&
        err.retryAfterMs===89000);
    time.advance(89000);
    assert.doesNotThrow(()=>throwIfAIProviderCoolingDown("signed doc"));
});
