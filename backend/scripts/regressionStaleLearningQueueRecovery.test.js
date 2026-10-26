import test from "node:test";
import assert from "node:assert/strict";
import {recoverStaleLearningQueue}
  from "../core/learning/learningQueueRecovery.js";

const recovered=[
  {id:"known-proposal",action:"NEW_SKILL",status:"PROPOSED"},
  {id:"retry-improvement",action:"SKILL_IMPROVEMENT",status:"PENDING"},
  {id:"ambiguous-candidate",action:"NEW_SKILL",status:"FAILED"},
  {id:"ambiguous-discovery",action:"PATTERN_DISCOVERY",status:"FAILED"}
];

test("stale recovery reports completed, safe retries and quarantines separately",async()=>{
  let params=null;
  const result=await recoverStaleLearningQueue({
    staleMinutes:1,limit:10000,
    client:{async rpc(name,payload){
      assert.equal(name,"recover_stale_learning_queue");
      params=payload;
      return {data:structuredClone(recovered),error:null};
    }}
  });
  assert.deepEqual(params,{p_stale_minutes:120,p_limit:100});
  assert.equal(result.success,true);
  assert.equal(result.recovered,4);
  assert.equal(result.provenComplete,1);
  assert.equal(result.retried,1);
  assert.equal(result.quarantined,2);
  assert.deepEqual(result.entries,recovered);
});

test("a failed database recovery cannot claim a safe retry or reinstate candidates",async()=>{
  const bad=await recoverStaleLearningQueue({client:{
    async rpc(){return {data:null,error:{message:"connection lost"}};}
  }});
  assert.equal(bad.success,false);
  assert.equal(bad.recovered,0);
  assert.match(bad.error,/connection lost/);
  assert.equal(bad.entries,undefined);
});

test("unexpected statuses cannot bypass fail-closed recovery contract",async()=>{
  const bad=await recoverStaleLearningQueue({client:{
    async rpc(){return {data:[{id:"x",status:"PROCESSING"}],error:null};}
  }});
  assert.equal(bad.success,false);
  assert.equal(bad.recovered,0);
  assert.match(bad.error,/Unexpected recovery transition/);
});

test("recovery returns zero when no leases are stale",async()=>{
  const result=await recoverStaleLearningQueue({client:{
    async rpc(){return {data:[],error:null};}
  }});
  assert.equal(result.success,true);
  assert.equal(result.recovered,0);
  assert.equal(result.retried,0);
  assert.equal(result.quarantined,0);
});

test("stale lease bounds are enforced for negative, oversized and nonnumeric values",async()=>{
  const cases=[
    {staleMinutes:-10,limit:-4,minutes:120,max:1},
    {staleMinutes:20000,limit:20000,minutes:10080,max:100},
    {staleMinutes:"invalid",limit:"invalid",minutes:120,max:100}
  ];
  for(const cfg of cases){
    const response=await recoverStaleLearningQueue({
      staleMinutes:cfg.staleMinutes,limit:cfg.limit,client:{
        async rpc(name,params){
          assert.equal(params.p_stale_minutes,cfg.minutes);
          assert.equal(params.p_limit,cfg.max);
          return {data:[],error:null};
        }
      }
    });
    assert.equal(response.success,true);
  }
});
