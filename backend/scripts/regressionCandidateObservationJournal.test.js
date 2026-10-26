import test from "node:test";
import assert from "node:assert/strict";
import {randomUUID} from "node:crypto";
import {applyCandidateObservation}
  from "../core/learning/candidateMemory/candidateObservationJournal.js";
import {observeLearningCandidate}
  from "../core/learning/learningCandidateMemory.js";

function atomicCandidateRpc(){
  const logs=new Map(),rows=new Map();
  let writes=0;
  const client={async rpc(name,p){
    assert.equal(name,"apply_candidate_memory_observation");
    if(!p.p_queue_item_id)return {data:null,error:{message:"Missing queue ID"}};
    if(logs.has(p.p_queue_item_id)){
      const candidateId=logs.get(p.p_queue_item_id);
      return {data:{existing:true,applied:false,candidate:structuredClone(rows.get(candidateId))},error:null};
    }
    if(p.p_action!=="NEW"){
      const row=rows.get(p.p_candidate_id);
      if(!row||row.occurrences!==p.p_expected_occurrences)
        return {data:null,error:{message:"Stale candidate observation"}};
    }
    const candidate=structuredClone(p.p_candidate);
    const row={
      id:p.p_candidate_id,candidate_key:p.p_candidate_key,action:"NEW_SKILL",
      skill_id:p.p_skill_id,category:p.p_category,status:"ACTIVE",
      confidence:Number(candidate.confidence||0),
      occurrences:Number(candidate.occurrences||1),
      candidate_json:candidate,metadata:structuredClone(p.p_metadata),
      created_at:"2026-10-27T00:00:00Z",updated_at:"2026-10-27T00:00:00Z"
    };
    logs.set(p.p_queue_item_id,p.p_candidate_id);
    rows.set(row.id,row);
    writes++;
    return {data:{existing:false,applied:true,candidate:structuredClone(row)},error:null};
  }};
  return {client,logs,rows,get writes(){return writes;}};
}
const candidate={skillId:"skill-probe",name:"Check source",examples:[],
  occurrences:1,confidence:0.6};
const args=(queueItemId,id,action="NEW",occ=1,expectedOccurrences=null)=>({
  queueItemId,id,candidateKey:"probe-key",action,
  candidate:{...candidate,occurrences:occ},
  metadata:{source:"test"},expectedOccurrences
});

test("same queue item is applied once, even under concurrent RPC calls",async()=>{
  const store=atomicCandidateRpc(),id=randomUUID();
  const tasks=Array.from({length:8},()=>
    applyCandidateObservation({...args("same-queue",id),client:store.client}));
  const results=await Promise.all(tasks);
  assert.equal(results.filter(x=>x.applied).length,1);
  assert.equal(results.filter(x=>x.existing).length,7);
  assert.equal(store.writes,1);
  assert.equal(store.logs.size,1);
  assert.equal(store.rows.get(id).occurrences,1);
});

test("one independent queue item increments once; repeating it cannot increase occurrences",async()=>{
  const store=atomicCandidateRpc(),id=randomUUID();
  await applyCandidateObservation({...args("create",id),client:store.client});
  const input={...args("merge",id,"EXACT",2,1),client:store.client};
  const first=await applyCandidateObservation(input);
  const second=await applyCandidateObservation(input);
  assert.equal(first.applied,true);
  assert.equal(second.existing,true);
  assert.equal(second.memory.occurrences,2);
  assert.equal(store.writes,2);
  assert.equal(store.logs.size,2);
  assert.equal(store.rows.get(id).occurrences,2);
});

test("two independent queue events with stale reads cannot overwrite newer evidence",async()=>{
  const store=atomicCandidateRpc(),id=randomUUID();
  await applyCandidateObservation({...args("create",id),client:store.client});
  await applyCandidateObservation({...args("merge-A",id,"EXACT",2,1),client:store.client});
  await assert.rejects(applyCandidateObservation({
    ...args("merge-B",id,"EXACT",2,1),client:store.client
  }),/Stale candidate observation/);
  assert.equal(store.rows.get(id).occurrences,2);
  assert.equal(store.logs.has("merge-B"),false);
});

test("NEW_SKILL observation uses atomic RPC instead of separate insert",async()=>{
  const store=atomicCandidateRpc();
  const item={id:"queue-memory-observe",action:"NEW_SKILL",
    event:{traceId:"trace-memory",payload:{skillCandidate:candidate}}};
  const observe=()=>observeLearningCandidate(item,{
    findCandidate:async()=>null,
    listCandidates:async()=>[],
    applyObservation:params=>applyCandidateObservation({...params,client:store.client})
  });
  const first=await observe(),again=await observe();
  assert.equal(first.success,true);
  assert.equal(first.observed,true);
  assert.equal(first.created,true);
  assert.equal(first.memory.candidateKey.length>0,true);
  assert.equal(again.success,true);
  assert.equal(again.observed,false);
  assert.equal(again.created,false);
  assert.equal(again.queueItem.event.payload.skillCandidate.occurrences,1);
  assert.equal(store.writes,1);
});

test("unverifiable queue identity never mutates candidate memory",async()=>{
  const store=atomicCandidateRpc();
  await assert.rejects(applyCandidateObservation({
    ...args("",randomUUID()),client:store.client
  }),/requires queue ID/);
  const failure=await applyCandidateObservation({
    ...args("query-failed",randomUUID()),client:{
      async rpc(){return {data:null,error:{message:"Supabase unavailable"}};}
    }
  }).then(()=>null,error=>error);
  assert.match(failure.message,/Supabase unavailable/);
  assert.equal(store.writes,0);
});
