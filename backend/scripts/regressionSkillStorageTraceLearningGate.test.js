import test from "node:test";
import assert from "node:assert/strict";
import {executeSubtask} from "../core/subtask/executeSubtask.js";
import {createJessicaExecutionTrace,recordExecutionResult,completeExecutionTrace}
  from "../core/trace/executionTraceFacade.js";
import {hasExperienceStorageIntegrityFailure}
  from "../core/learning/experienceStorageIntegrityGate.js";
import {shouldAnalyzeLearningTrace}
  from "../core/learning/trigger/triggerEligibility.js";
import {processExecutionLearning} from "../core/learning/learningFacade.js";

const subtask={
  id:"failing-supabase-skill",
  text:"Verify a document using a stored Skill",
  memoryHints:[{id:"stale-old-version",name:"Cached Skill",
    workflow:["unsafe outdated step"]}]
};
function traceResult(result){
  const trace=createJessicaExecutionTrace(subtask.text);
  recordExecutionResult(trace,result);
  completeExecutionTrace(trace);
  return trace;
}

test("unverified Skill storage is reported in outer Trace and blocks Learning",async()=>{
  const result=await executeSubtask(subtask,{experienceResolver:async()=>({
    found:false,reason:"storage-error",experience:null
  })});
  assert.equal(result.success,false);
  assert.equal(result.status,"FAILED");
  assert.equal(result.stage,"experience-storage");
  assert.equal(result.failure?.type,"EXPERIENCE_STORAGE_UNVERIFIED");
  assert.equal(result.executionMeta?.executionStarted,false);
  assert.deepEqual(result.experience?.skills,[]);
  const trace=traceResult(result);
  assert.equal(trace.status,"FAILED");
  assert.ok(trace.events.some(event=>event.type==="RESULT"));
  assert.equal(trace.failures.length,1);
  assert.equal(trace.failures[0].type,"EXPERIENCE_STORAGE_UNVERIFIED");
  assert.equal(trace.experienceUsage.used,false);
  assert.deepEqual(trace.experienceUsage.skills,[]);
  assert.equal(hasExperienceStorageIntegrityFailure(trace),true);
  assert.equal(shouldAnalyzeLearningTrace(trace),false);
  const learning=await processExecutionLearning(trace);
  assert.equal(learning.success,false);
  assert.equal(learning.queued,false);
  assert.equal(learning.skipped,true);
  assert.equal(learning.reason,"experience_storage_integrity_failure");
});

test("thrown resolver error also fails closed before tools or AI planner",async()=>{
  const result=await executeSubtask(subtask,{experienceResolver:async()=>{
    throw new Error("mocked Supabase module query failed");
  }});
  assert.equal(result.stage,"experience-storage");
  assert.equal(result.failure?.type,"EXPERIENCE_STORAGE_UNVERIFIED");
  const trace=traceResult(result);
  assert.equal(trace.failures.length,1);
  assert.equal((await processExecutionLearning(trace)).queued,false);
});

test("complex trace cannot learn when any child lacks verified Skill storage",async()=>{
  const trace=createJessicaExecutionTrace("two tasks");
  trace.result={
    success:true,status:"COMPLETED",
    results:[
      {success:true,status:"COMPLETED",verified:true,validation:{valid:true}},
      {success:true,status:"COMPLETED",verified:true,validation:{valid:true},
       failure:{type:"EXPERIENCE_STORAGE_UNVERIFIED"}}
    ]
  };
  trace.statistics.completed=2;
  trace.status="COMPLETED";
  assert.equal(hasExperienceStorageIntegrityFailure(trace),true);
  assert.equal(shouldAnalyzeLearningTrace(trace),false);
  assert.equal((await processExecutionLearning(trace)).queued,false);
});

test("storage marker veto remains effective even with optimistic completion",async()=>{
  const trace=createJessicaExecutionTrace("optimistic-result");
  trace.result={
    success:true,status:"COMPLETED",verified:true,validation:{valid:true},
    executionMeta:{experienceIntegrityFailure:true}
  };
  trace.status="COMPLETED";
  trace.statistics.completed=1;
  assert.equal(shouldAnalyzeLearningTrace(trace),false);
  const learning=await processExecutionLearning(trace);
  assert.equal(learning.reason,"experience_storage_integrity_failure");
});
