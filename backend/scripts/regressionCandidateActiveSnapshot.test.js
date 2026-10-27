import test from "node:test";
import assert from "node:assert/strict";
import {updateCandidateMemory}
  from "../core/learning/candidateMemory/candidateStorage.js";

const ID="00000000-0000-4000-8000-000000000501";
const REV="2026-10-26T12:00:00.000Z";
const evidence={skillId:"stable-skill",confidence:0.83,occurrences:2,
  examples:[{traceId:"verified-trace"}]};

function database(status="ACTIVE"){
  const state={id:ID,status,updated_at:REV,revision:0,metadata:{origin:"verified"},
    candidate_json:{...evidence,occurrences:1},occurrences:1,confidence:0.72};
  let writes=0;
  return {
    state,
    get writes(){return writes;},
    client:{from(table){
      assert.equal(table,"learning_candidates");
      const filters={};
      let payload=null;
      return {
        update(data){payload=data;return this;},
        eq(key,value){filters[key]=value;return this;},
        select(){return this;},
        async maybeSingle(){
          if(Object.entries(filters).some(([k,v])=>state[k]!==v))
            return {data:null,error:null};
          Object.assign(state,payload);
          state.revision++;
          writes++;
          return {data:structuredClone(state),error:null};
        }
      };
    }}
  };
}

function mutate(db,expectedUpdatedAt=REV,expectedRevision=0){
  return updateCandidateMemory({
    id:ID,candidate:evidence,skillId:evidence.skillId,
    metadata:{origin:"merged"},
    expectedUpdatedAt,expectedRevision,client:db.client
  });
}

test("ACTIVE Candidate evidence can be updated once with a valid revision",async()=>{
  const db=database();
  const updated=await mutate(db);
  assert.equal(updated.status,"ACTIVE");
  assert.equal(updated.occurrences,2);
  assert.equal(db.writes,1);
  assert.equal(db.state.metadata.origin,"merged");
});

for(const status of ["PROMOTED","REJECTED","MERGED"]){
  test(status+" Candidate is immutable to consolidation evidence writes",async()=>{
    const db=database(status);
    const before=structuredClone(db.state);
    await assert.rejects(mutate(db),/changed or finalized/);
    assert.equal(db.writes,0);
    assert.deepEqual(db.state,before);
  });
}

test("stale ACTIVE snapshot cannot overwrite fresher observation",async()=>{
  const db=database();
  db.state.updated_at="2026-10-26T12:00:01.000Z";
  const before=structuredClone(db.state);
  await assert.rejects(mutate(db),/changed or finalized/);
  assert.equal(db.writes,0);
  assert.deepEqual(db.state,before);
});

test("parallel updates of one Candidate revision cannot both succeed",async()=>{
  const db=database();
  const outcomes=await Promise.allSettled([mutate(db),mutate(db)]);
  assert.deepEqual(outcomes.map(x=>x.status).sort(),["fulfilled","rejected"]);
  assert.equal(db.writes,1);
});

test("missing snapshot proof fails before querying Supabase",async()=>{
  const db=database();
  await assert.rejects(updateCandidateMemory({id:ID,candidate:evidence,client:db.client}),/ACTIVE Candidate revision proof required/);
  assert.equal(db.writes,0);
});

test("database error never reports Candidate evidence write success",async()=>{
  const client={from(){
    return {update(){return this;},eq(){return this;},select(){return this;},
      async maybeSingle(){return {data:null,error:{message:"database offline"}};}}
  }};
  await assert.rejects(updateCandidateMemory({
    id:ID,candidate:evidence,expectedUpdatedAt:REV,expectedRevision:0,client
  }),/database offline/);
});

test("unchanged updated_at cannot revive a stale revision",async()=>{
  const db=database();
  db.state.revision=1; // concurrent owned observation committed within same millisecond
  const before=structuredClone(db.state);
  await assert.rejects(mutate(db,REV,0),/changed or finalized/);
  assert.equal(db.writes,0);
  assert.deepEqual(db.state,before);
});

test("malformed revision fails closed before querying the client",async()=>{
  const db=database();
  await assert.rejects(mutate(db,REV,undefined),/revision proof required/);
  assert.equal(db.writes,0);
});
