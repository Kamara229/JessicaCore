import test from "node:test";
import assert from "node:assert/strict";
import {loadExperiences} from "../experience/storage/supabaseExperienceStore.js";
import {createJessicaExecutionTrace,recordExecutionSummary,completeExecutionTrace}
  from "../core/trace/executionTraceFacade.js";
import {runLearningTrigger} from "../core/learning/learningTrigger.js";
import {createLearningQueueItem}
  from "../core/learning/learningQueue.js";
import {persistLearningEvents}
  from "../core/learning/coordinator/learningQueueBatch.js";
import {saveLearningQueueItem}
  from "../core/learning/learningQueueStorage.js";
import {modularExperienceClient,v1Id,v2Id}
  from "./fixtures/verifiedModularSkills.js";
import {learningQueueInsertOnlyClient}
  from "./fixtures/learningQueueInsertOnlyClient.js";

async function confirmedEvents(){
  const skills=await loadExperiences(modularExperienceClient());
  const children=skills.map(skill=>({
    id:"subtask-"+skill.id,
    task:"Verify source and use the selected Skill",
    status:"COMPLETED",success:true,verified:true,
    validation:{valid:true,degraded:false,outcomeType:"result"},
    experience:{found:true,used:true,applied:true,
      appliedSkillIds:[skill.id],skills:[skill]}
  }));
  const trace=createJessicaExecutionTrace("Two verified independent Skills");
  recordExecutionSummary(trace,{status:"COMPLETED",success:true,results:children});
  completeExecutionTrace(trace);
  const first=runLearningTrigger(trace);
  assert.equal(first.triggered,true,first.error);
  assert.deepEqual(new Set(first.learningEvents.map(e=>e.skillId)),
    new Set([v1Id,v2Id]));
  return {trace,first};
}

test("duplicate processing of two Skills queues exactly one item per Skill",async()=>{
  const {trace,first}=await confirmedEvents();
  const store=learningQueueInsertOnlyClient();
  const saveQueueItem=item=>saveLearningQueueItem(item,{client:store.client});
  const one=await persistLearningEvents(first.learningEvents,{saveQueueItem});
  assert.equal(one.success,true);
  assert.equal(one.queuedCount,2);
  assert.equal(one.existingCount,0);
  assert.equal(store.rows.size,2);
  assert.equal(new Set(one.queueItems.map(q=>q.id)).size,2);
  assert.deepEqual(new Set(one.queueItems.map(q=>q.skillId)),
    new Set([v1Id,v2Id]));
  for(const row of store.rows.values()){
    row.status="PROPOSED";
    row.processed_at="2026-10-27T10:00:00Z";
  }
  const replay=runLearningTrigger(trace); // event timestamps change, IDs must not
  const two=await persistLearningEvents(replay.learningEvents,{saveQueueItem});
  assert.equal(two.success,true);
  assert.equal(two.queuedCount,0);
  assert.equal(two.existingCount,2);
  assert.equal(two.failedCount,0);
  assert.equal(two.queueItems.length,0);
  assert.equal(store.rows.size,2);
  assert.deepEqual(two.batch.map(x=>x.queueItem.id),
    one.batch.map(x=>x.queueItem.id));
  for(const row of store.rows.values()){
    assert.equal(row.status,"PROPOSED","replay cannot reopen processed Skill");
    assert.equal(row.processed_at,"2026-10-27T10:00:00Z");
  }
  assert.equal(store.calls.length,4);
  assert.ok(store.calls.every(c=>c.options.onConflict==="id" &&
    c.options.ignoreDuplicates===true));
});

test("concurrent replays deduplicate using one stable unique identifier",async()=>{
  const {first}=await confirmedEvents();
  const store=learningQueueInsertOnlyClient();
  const saveQueueItem=item=>saveLearningQueueItem(item,{client:store.client});
  const [left,right]=await Promise.all([
    persistLearningEvents(first.learningEvents,{saveQueueItem}),
    persistLearningEvents(structuredClone(first.learningEvents),{saveQueueItem})
  ]);
  assert.equal(left.success,true);
  assert.equal(right.success,true);
  assert.equal(left.queuedCount+right.queuedCount,2);
  assert.equal(left.existingCount+right.existingCount,2);
  assert.equal(store.rows.size,2);
});

test("different Skills and Traces retain independent stable queue identities",async()=>{
  const {trace,first}=await confirmedEvents();
  const original=first.learningEvents.map(createLearningQueueItem);
  const copy=runLearningTrigger(trace).learningEvents.map(createLearningQueueItem);
  assert.deepEqual(copy.map(i=>i.id),original.map(i=>i.id));
  assert.equal(new Set(original.map(i=>i.id)).size,2);
  const newTrace=structuredClone(first.learningEvents);
  newTrace.forEach(event=>{event.traceId="separate-trace-id";});
  assert.ok(newTrace.every((event,i)=>
    createLearningQueueItem(event).id!==original[i].id));
});
