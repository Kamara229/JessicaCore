import test from "node:test";
import assert from "node:assert/strict";
import {transitionCandidateFinalStatus}
  from "../core/learning/candidateMemory/candidateFinalStatus.js";

const id="00000000-0000-4000-8000-000000000101";
function store(){
  let row={id,status:"ACTIVE",updated_at:"2026-10-27T00:00:00Z",revision:0,
    metadata:{source:"verified-execution"},occurrences:2};
  let writes=0,reads=0;
  const client={from(table){
    assert.equal(table,"learning_candidates");
    const query={mode:"read",filters:{},payload:null,
      select(){return this;},
      update(value){this.mode="update";this.payload=value;return this;},
      eq(key,value){this.filters[key]=value;return this;},
      async maybeSingle(){
        if(this.mode==="read"){
          reads++;
          return {data:row?.id===this.filters.id?structuredClone(row):null,error:null};
        }
        const matches=row&&Object.entries(this.filters)
          .every(([key,value])=>row[key]===value);
        if(!matches)return {data:null,error:null};
        row={...row,...this.payload,revision:row.revision+1};
        writes++;
        return {data:structuredClone(row),error:null};
      }
    };
    return query;
  }};
  return {client,get row(){return structuredClone(row);},
    get writes(){return writes;},get reads(){return reads;}};
}
function promote(db,proposalId="proposal-one",resolvedAt="first"){
  return transitionCandidateFinalStatus({client:db.client,id,status:"PROMOTED",
    metadata:{proposalId,skillId:"skill-one",resolvedAt}});
}
test("PROMOTED replay is read-only and preserves the original metadata",async()=>{
  const db=store();
  const first=await promote(db);
  const initial=structuredClone(db.row);
  const replay=await promote(db,"proposal-one","retry");
  assert.equal(first.success,true);
  assert.equal(first.updated,true);
  assert.equal(replay.success,true);
  assert.equal(replay.existing,true);
  assert.equal(replay.updated,false);
  assert.equal(db.writes,1);
  assert.deepEqual(db.row,initial);
  assert.equal(db.row.occurrences,2);
});
test("different Proposal cannot relabel an already PROMOTED candidate",async()=>{
  const db=store();
  await promote(db);
  const blocked=await promote(db,"proposal-two");
  assert.equal(blocked.success,false);
  assert.equal(blocked.conflict,true);
  assert.equal(db.writes,1);
  assert.equal(db.row.metadata.proposalId,"proposal-one");
});
test("late REJECTED decision cannot demote PROMOTED Candidate",async()=>{
  const db=store();
  await promote(db);
  const rejected=await transitionCandidateFinalStatus({
    client:db.client,id,status:"REJECTED",metadata:{proposalId:"proposal-one"}
  });
  assert.equal(rejected.success,false);
  assert.equal(rejected.conflict,true);
  assert.equal(db.row.status,"PROMOTED");
});
test("two concurrent promotion requests commit exactly once",async()=>{
  const db=store();
  const [a,b]=await Promise.all([promote(db),promote(db)]);
  assert.equal(a.success,true);
  assert.equal(b.success,true);
  assert.deepEqual([a.updated,b.updated].sort(),[false,true]);
  assert.equal(db.writes,1);
});
test("MERGED retirement is idempotent and may not override PROMOTED",async()=>{
  const db=store();
  const args={client:db.client,id,status:"MERGED",
    metadata:{mergedIntoId:"canonical-one",mergedAt:"first"}};
  const first=await transitionCandidateFinalStatus(args);
  const replay=await transitionCandidateFinalStatus({
    ...args,metadata:{...args.metadata,mergedAt:"second"}
  });
  assert.equal(first.updated,true);
  assert.equal(replay.existing,true);
  assert.equal(db.writes,1);
  const conflict=await promote(db);
  assert.equal(conflict.success,false);
  assert.equal(db.row.status,"MERGED");
});
test("database failures and missing decision identities fail closed",async()=>{
  const badClient={from(){return {select(){return this;},eq(){return this;},
    async maybeSingle(){return {data:null,error:{message:"DB unavailable"}};}};}};
  const failed=await transitionCandidateFinalStatus({client:badClient,id,
    status:"PROMOTED",metadata:{proposalId:"proposal-one"}});
  assert.equal(failed.success,false);
  assert.match(failed.error,/DB unavailable/);
  const db=store();
  const missing=await transitionCandidateFinalStatus({client:db.client,id,
    status:"PROMOTED",metadata:{skillId:"skill-one"}});
  assert.equal(missing.success,false);
  assert.equal(db.writes,0);
});

test("finalization cannot commit over same-timestamp newer Candidate revision",async()=>{
  const db=store();
  // Simulates an observation that changed the row in the same millisecond.
  const current=await db.client.from("learning_candidates").select("*")
    .eq("id",id).maybeSingle();
  assert.equal(current.data.revision,0);
  // An independent writer has bumped the revision without changing updated_at.
  const edited=await db.client.from("learning_candidates").update({
    occurrences:3
  }).eq("id",id).eq("revision",0).select("*").maybeSingle();
  assert.equal(edited.data.revision,1);
  assert.equal(edited.data.updated_at,current.data.updated_at);
  const final=await promote(db);
  assert.equal(final.success,true);
  assert.equal(final.candidate.revision,2);
  assert.equal(final.candidate.occurrences,3);
});
