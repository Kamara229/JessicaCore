import test from "node:test";
import assert from "node:assert/strict";
import {loadExperiences} from "../experience/storage/supabaseExperienceStore.js";
import {buildExecutionExperienceContext}
  from "../core/subtask/subtaskExperienceContext.js";
import {createJessicaExecutionTrace,recordExecutionResult,completeExecutionTrace}
  from "../core/trace/executionTraceFacade.js";
import {shouldAnalyzeLearningTrace}
  from "../core/learning/trigger/triggerEligibility.js";
import {runLearningTrigger} from "../core/learning/learningTrigger.js";
import {modularExperienceClient,v1Id,v2Id}
  from "./fixtures/verifiedModularSkills.js";

function createVerifiedTrace(skill,overrides={}){
  const result={
    status:"COMPLETED",success:true,verified:true,
    validation:{valid:true,degraded:false,outcomeType:"result"},
    experience:{
      found:true,used:true,applied:true,appliedSkillIds:[skill.id],
      source:"experience-search",skills:[skill]
    },
    ...overrides
  };
  const trace=createJessicaExecutionTrace("Verify a source using stored Skill");
  recordExecutionResult(trace,result);
  completeExecutionTrace(trace);
  return trace;
}

for(const skillId of [v1Id,v2Id]){
  test("verified "+skillId+" reaches Execution Trace and Learning improvement",async()=>{
    const skills=await loadExperiences(modularExperienceClient());
    const selected=skills.find(s=>s.id===skillId);
    assert.ok(selected);
    assert.deepEqual(selected.workflow,["read source","verify output"]);
    assert.equal(selected.moduleManifest,undefined);
    const execution=buildExecutionExperienceContext({
      found:true,source:"experience-search",confidence:0.96,
      experience:selected
    });
    assert.equal(execution.skills.length,1);
    assert.equal(execution.skills[0].id,skillId);
    const trace=createVerifiedTrace(execution.skills[0]);
    assert.equal(trace.status,"COMPLETED");
    assert.equal(trace.experienceUsage.used,true);
    assert.equal(trace.experienceUsage.applied,true);
    assert.deepEqual(trace.experienceUsage.appliedSkillIds,[skillId]);
    assert.deepEqual(trace.experienceUsage.skillIds,[skillId]);
    assert.equal(shouldAnalyzeLearningTrace(trace),true);
    const trigger=runLearningTrigger(trace);
    assert.equal(trigger.triggered,true,trigger.reason||trigger.error);
    assert.equal(trigger.learningEvents.length,1);
    assert.equal(trigger.learningEvents[0].action,"SKILL_IMPROVEMENT");
    assert.equal(trigger.learningEvents[0].skillId,skillId);
    assert.equal(trigger.analyses[0].metadata.positiveReinforcementAllowed,true);
  });

  test("selected "+skillId+" is not reinforced without proof of application",async()=>{
    const skill=(await loadExperiences(modularExperienceClient()))
      .find(x=>x.id===skillId);
    for(const experience of [
      {found:true,used:true,applied:false,appliedSkillIds:[],skills:[skill]},
      {found:true,used:true,applied:true,appliedSkillIds:["unknown-skill"],skills:[skill]},
      {found:true,used:false,applied:false,appliedSkillIds:[],skills:[skill]}
    ]){
      const trace=createVerifiedTrace(skill,{experience});
      assert.notEqual(trace.experienceUsage.applied,true);
      const trigger=runLearningTrigger(trace);
      assert.equal(trigger.learningEvents.some(e=>e.action==="SKILL_IMPROVEMENT"),
        false,"a found but unapplied Skill must never be strengthened");
    }
  });

  test("applied "+skillId+" without verified independent result is not reinforced",async()=>{
    const skill=(await loadExperiences(modularExperienceClient()))
      .find(x=>x.id===skillId);
    for(const resultOverrides of [
      {verified:false},
      {validation:{valid:false}},
      {validation:{valid:true,degraded:true}},
      {validation:{valid:true,outcomeType:"no_verified_result"}},
      {status:"NO_VERIFIED_RESULT",verified:false,success:false}
    ]){
      const trace=createVerifiedTrace(skill,resultOverrides);
      assert.equal(trace.experienceUsage.applied,true);
      assert.equal(shouldAnalyzeLearningTrace(trace),false);
      const trigger=runLearningTrigger(trace);
      assert.equal(trigger.learningEvents.some(e=>e.action==="SKILL_IMPROVEMENT"),
        false,"unverified outcomes cannot reinforce a Skill");
    }
  });
}
