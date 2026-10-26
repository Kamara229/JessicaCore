import test from "node:test";
import assert from "node:assert/strict";
import {loadExperiences} from "../experience/storage/supabaseExperienceStore.js";
import {createJessicaExecutionTrace,recordExecutionSummary,completeExecutionTrace}
  from "../core/trace/executionTraceFacade.js";
import {runLearningTrigger} from "../core/learning/learningTrigger.js";
import {shouldAnalyzeLearningTrace}
  from "../core/learning/trigger/triggerEligibility.js";
import {modularExperienceClient,v1Id,v2Id}
  from "./fixtures/verifiedModularSkills.js";

async function skills(){
  const rows=await loadExperiences(modularExperienceClient());
  return [rows.find(s=>s.id===v1Id),rows.find(s=>s.id===v2Id)];
}
function child(skill,{applied=true,verified=true,degraded=false,valid=true}={}){
  return {
    id:"child-"+skill.id,task:"Use independently confirmed skill",
    status:"COMPLETED",success:true,verified,
    validation:{valid,degraded,outcomeType:"result"},
    experience:{
      found:true,used:true,applied,
      appliedSkillIds:applied?[skill.id]:[],
      skills:[skill],source:"experience-search",confidence:0.94
    },
    executionMeta:{usedTools:["local_verifier"]}
  };
}
function outerTrace(children,summaryExperience=null){
  const trace=createJessicaExecutionTrace("Verify both independent skills");
  recordExecutionSummary(trace,{
    status:"COMPLETED",success:true,
    results:children,
    ...(summaryExperience?{experience:summaryExperience}:{})
  });
  completeExecutionTrace(trace);
  return trace;
}
function improvements(trace){
  const result=runLearningTrigger(trace);
  assert.equal(result.error,undefined,result.error);
  return result.learningEvents.filter(x=>x.action==="SKILL_IMPROVEMENT");
}
test("v1 and v2 independently applied verified children each reinforce once",async()=>{
  const [v1,v2]=await skills();
  assert.ok(v1&&v2);
  assert.deepEqual(v1.workflow,["read source","verify output"]);
  assert.deepEqual(v2.workflow,["read source","verify output"]);
  const trace=outerTrace([child(v1),child(v2)]);
  assert.equal(trace.status,"COMPLETED");
  assert.equal(trace.experienceUsage.applied,true);
  assert.deepEqual(new Set(trace.experienceUsage.appliedSkillIds),
    new Set([v1Id,v2Id]));
  assert.equal(shouldAnalyzeLearningTrace(trace),true);
  const events=improvements(trace);
  assert.equal(events.length,2);
  assert.deepEqual(new Set(events.map(x=>x.skillId)),new Set([v1Id,v2Id]));
  assert.ok(events.every(e=>e.reusable===true));
});
test("found but unapplied v2 cannot be reinforced from verified sibling v1",async()=>{
  const [v1,v2]=await skills();
  const trace=outerTrace([child(v1),child(v2,{applied:false})]);
  assert.equal(shouldAnalyzeLearningTrace(trace),true);
  const result=runLearningTrigger(trace);
  const events=result.learningEvents.filter(x=>x.action==="SKILL_IMPROVEMENT");
  assert.deepEqual(events.map(e=>e.skillId),[v1Id]);
  assert.equal(events.some(e=>e.skillId===v2Id),false);
  assert.equal(trace.experienceUsage.skills.length,2);
  assert.deepEqual(trace.experienceUsage.appliedSkillIds,[v1Id]);
});
test("parent-only applied attribution never substitutes child proof",async()=>{
  const [v1,v2]=await skills();
  const summaryExperience={
    found:true,used:true,applied:true,appliedSkillIds:[v1Id,v2Id],
    skills:[v1,v2]
  };
  const trace=outerTrace([child(v1),child(v2,{applied:false})],
    summaryExperience);
  assert.equal(trace.experienceUsage.appliedSkillIds.includes(v2Id),true);
  const events=improvements(trace);
  assert.deepEqual(events.map(e=>e.skillId),[v1Id],
    "parent-only attribution must not reinforce v2");
});
test("unverified/degraded child vetoes reinforcement despite optimistic parent",async()=>{
  const [v1,v2]=await skills();
  for(const options of [
    {verified:false},
    {valid:false},
    {degraded:true}
  ]){
    const trace=outerTrace([child(v1),child(v2,options)]);
    assert.equal(shouldAnalyzeLearningTrace(trace),false);
    assert.equal(improvements(trace).length,0);
  }
});
test("unapplied v1 and applied v2 improve only v2",async()=>{
  const [v1,v2]=await skills();
  const trace=outerTrace([child(v1,{applied:false}),child(v2)]);
  assert.equal(shouldAnalyzeLearningTrace(trace),true);
  assert.deepEqual(improvements(trace).map(e=>e.skillId),[v2Id]);
});
