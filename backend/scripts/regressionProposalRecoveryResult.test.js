import test from "node:test";
import assert from "node:assert/strict";
import {recoverStaleLearningProposals}
  from "../core/learning/proposalStorage/proposalStorageRecovery.js";

function mock(result,error=null){
  let calls=0;
  const client={async rpc(name,args){
    calls++;
    assert.equal(name,"recover_stale_learning_proposals");
    assert.deepEqual(args,{p_stale_minutes:30,p_limit:8});
    return {data:result,error};
  }};
  return {client,get calls(){return calls;}};
}

test("valid recovery count is reported without additional SQL calls",async()=>{
  for(const data of [{recovered:3},[{recovered:0}],{recovered:8}]){
    const db=mock(data);
    const result=await recoverStaleLearningProposals({
      client:db.client,staleMinutes:30,limit:8
    });
    assert.equal(result.success,true);
    assert.equal(result.recovered,Array.isArray(data)?
      data[0].recovered:data.recovered);
    assert.equal(db.calls,1);
  }
});

test("malformed recovery responses never report success",async()=>{
  for(const data of [null,[],[{recovered:1},{recovered:2}],{},
    {recovered:"3"},{recovered:-1},{recovered:1.5},
    {recovered:9},{recovered:NaN}]){
    const db=mock(data);
    const result=await recoverStaleLearningProposals({
      client:db.client,staleMinutes:30,limit:8
    });
    assert.equal(result.success,false);
    assert.equal(result.recovered,0);
    assert.match(result.error,/Invalid Proposal recovery/);
    assert.equal(db.calls,1);
  }
});

test("database error leaves recovery unverified without retries",async()=>{
  const db=mock(null,{message:"connection unavailable"});
  const result=await recoverStaleLearningProposals({
    client:db.client,staleMinutes:30,limit:8
  });
  assert.equal(result.success,false);
  assert.equal(result.recovered,0);
  assert.match(result.error,/connection unavailable/);
  assert.equal(db.calls,1);
});
