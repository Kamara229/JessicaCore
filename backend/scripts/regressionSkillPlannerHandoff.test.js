import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {searchExperience} from "../experience/search/experienceSearch.js";
import {buildExperienceContext} from "../experience/context/experienceContext.js";
import {buildExecutionExperienceContext} from "../core/subtask/subtaskExperienceContext.js";
import {normalizePlanningContext} from "../core/planner/planningContext.js";
import {buildPlanningContextText} from "../core/planner/planningContextText.js";
import {tryCreateKnownLicenseComparison} from "../core/planner/knownLicenseComparison.js";
import {createExecutionContext} from "../core/execution/context/contextFactory.js";

const task="Сравни патентные условия MIT License и Apache 2.0 по официальным текстам. Есть ли явная патентная лицензия и когда она прекращается? Укажи пункты и ссылки.";
const skill={
 id:"license-patent-clause-comparison",name:"License Patent Clause Comparison",
 version:1,status:"published",enabled:true,confidence:.82,
 triggerPatterns:["Compare patent clauses of license A and license B",
  "Identify termination conditions for patent rights in a license"],
 keywords:["patent conditions","patent grants"],
 workflow:["Identify official URLs for each license.",
  "Fetch both complete original license texts.",
  "Check citations against the original document."],
 sourcePriority:["Official legal license texts only."],
 validationRules:["Never invent a patent grant not expressed in the source."],
 examples:[{traceId:"verified-procedure",result:{verified:true,success:true}}],
 learning:{maturityLevel:"LEARNING",confidence:.82},
 metadata:{learningMode:"autonomous"}
};
function chain(){
 const result=searchExperience(task,[skill,{...skill,id:"draft",enabled:false}]);
 assert.equal(result.found,true);
 assert.equal(result.experience.id,skill.id);
 const planner=buildExperienceContext(result);
 const execution=buildExecutionExperienceContext({
  found:true,experience:result.experience,planningContext:planner,
  source:"experience-search",confidence:result.confidence
 });
 return {result,planner,execution};
}
test("published Skill identity, version and evidence survive planner/executor handoff",()=>{
 const {planner,execution}=chain();
 assert.equal(planner.metadata.skillId,skill.id);
 assert.equal(planner.metadata.skillVersion,1);
 assert.equal(execution.skills[0].examples[0].traceId,"verified-procedure");
 assert.equal(execution.skills[0].id,skill.id);
});
test("selected skill workflow and source priority appear in AI Planner text",()=>{
 const {planner}=chain();
 const normalized=normalizePlanningContext(planner);
 assert.deepEqual(normalized.experience.workflow,skill.workflow);
 assert.deepEqual(normalized.sourceRules,skill.sourcePriority);
 const prompt=buildPlanningContextText(planner);
 assert.match(prompt,/Fetch both complete original license texts/);
 assert.match(prompt,/Official legal license texts only/);
 assert.match(prompt,/Never invent a patent grant/);
 assert.match(prompt,/License Patent Clause Comparison/);
});
test("native canonical route remains grounded and does not falsely claim skill execution",()=>{
 const {planner,execution}=chain();
 const native=tryCreateKnownLicenseComparison({task,planningContext:planner});
 assert.equal(native.handled,true);
 assert.equal(native.plan.evidence.mode,"source_content");
 assert.equal(native.plan.steps.length,2);
 assert.deepEqual(native.plan.steps.map(x=>x.tool),["web_fetch","web_fetch"]);
 const cycle=createExecutionContext({
   task,plan:native.plan,planningContext:planner,
   experience:{...execution,used:false}
 });
 assert.equal(cycle.planningContext.experience.id,skill.id);
 assert.equal(cycle.experience.found,true);
 assert.equal(cycle.experience.used,false);
 assert.equal(cycle.experience.skills[0].id,skill.id);
});


test("verified date Skill executes its real native date_diff step, not just the Planner DTO",async()=>{
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {executeTaskRunner}=await import("../core/execution/stepExecution/taskRunnerExecutor.js");
 await import("../tools/dateDiffTool.js");
 const verified={
   id:"date-difference-calculation",version:3,status:"published",enabled:true,
   confidence:0.82,learning:{maturityLevel:"NEW",confidence:0.82,successCount:1},
   executionProgram:{version:1,mode:"native",strategy:"date_difference",evidenceMode:"none"},
   examples:[{success:true,executedTools:["date_diff"],result:{
      success:true,verified:true,status:"COMPLETED",
      validation:{valid:true,degraded:false}
   }}]
 };
 const userTask="Рассчитай, сколько полных дней прошло до 18 октября 2026 года";
 const chosen=buildExecutionExperienceContext({
   found:true,source:"experience-search",confidence:0.92,experience:verified,
   planningContext:{experience:{id:verified.id}}
 });
 assert.equal(chosen.skills[0].id,verified.id);
 const planned=tryCreateNativeSkillPlan({
   task:userTask,dependencyContext:"Дата выпуска 2026-10-09",
   executionExperience:chosen,planningContext:{}
 });
 assert.equal(planned.handled,true);
 assert.equal(planned.source,"experience:date-difference-calculation");
 assert.equal(planned.plan.experience.used,true);
 assert.equal(planned.plan.steps[0].tool,"date_diff");
 const execution=createExecutionContext({
   task:userTask,plan:planned.plan,planningContext:{},
   experience:chosen
 });
 const actual=await executeTaskRunner(execution);
 assert.equal(actual.success,true,JSON.stringify(actual.failure||{}));
 assert.equal(execution.experience.used,true);
 assert.equal(execution.experience.skills[0].id,verified.id);
 assert.equal(execution.runResult?.success,true);
 assert.equal(execution.runResult?.results?.[0]?.data?.fullDays,9);
 const {syncTraceFromContext}=await import("../core/trace/traceContextAdapter.js");
 const {createExecutionTrace}=await import("../core/trace/traceFactory.js");
 const actualTrace=createExecutionTrace(userTask);
 syncTraceFromContext(actualTrace,execution);
 assert.equal(actualTrace.experienceUsage.applied,true);
 assert.deepEqual(actualTrace.experienceUsage.appliedSkillIds,[verified.id]);
 assert.deepEqual(actualTrace.experienceUsage.appliedStepIds,["native_date_diff"]);
 assert.equal(actualTrace.events.some(x=>x.type==="SKILL_APPLIED"),true);
});

test("unverified date evidence triggers native capability but not false Skill use",async()=>{
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const candidate={
   id:"date-difference-calculation",enabled:true,
   learning:{maturityLevel:"NEW",confidence:.8,successCount:1},
   executionProgram:{version:1,mode:"native",strategy:"date_difference",evidenceMode:"none"},
   examples:[{success:true,executedTools:["date_diff"],result:{
     success:true,verified:false,status:"COMPLETED",validation:{valid:true}
   }}]
 };
 const actual=tryCreateNativeSkillPlan({
  task:"Рассчитай, сколько полных дней прошло до 18 октября 2026 года",
  dependencyContext:"Дата выпуска 2026-10-09",
  executionExperience:{skills:[candidate]},planningContext:{}
 });
 assert.equal(actual.handled,true);
 assert.equal(actual.source,"native-date-diff");
 assert.equal(actual.plan.experience.used,false);
});


test("Execution Trace proves native Skill application from successful step results",async()=>{
 const {inspectAppliedSkill}=await import("../core/trace/traceSkillApplication.js");
 const {syncTraceFromContext}=await import("../core/trace/traceContextAdapter.js");
 const {createExecutionTrace}=await import("../core/trace/traceFactory.js");
 const p={
   experience:{used:true,skills:[{id:"date-difference-calculation"}]},
   steps:[{id:"native_date_diff",tool:"date_diff",experienceUsed:true}]
 };
 const full={
    experience:{used:true,found:true,skills:[{id:"date-difference-calculation"}]},
    plan:p,runResult:{success:true,results:[{id:"native_date_diff",tool:"date_diff",success:true,data:{fullDays:9}}]}
 };
 assert.equal(inspectAppliedSkill(full).applied,true);
 const trace=createExecutionTrace("Calculate days");
 syncTraceFromContext(trace,full);
 assert.equal(trace.experienceUsage.applied,true);
 assert.deepEqual(trace.experienceUsage.appliedSkillIds,["date-difference-calculation"]);
 assert.equal(trace.events.filter(x=>x.type==="SKILL_APPLIED").length,1);
 for(const variant of [
   {...full,runResult:{success:true,results:[]}},
   {...full,runResult:{success:true,results:[{id:"native_date_diff",tool:"date_diff",success:false}]}},
   {...full,experience:{...full.experience,used:false}},
   {...full,plan:{...p,experience:{used:false,skills:p.experience.skills}}},
   {...full,plan:{...p,experience:{used:true,skills:[{id:"different-skill"}]}}},
   {...full,runResult:{success:false,results:full.runResult.results}}
 ])assert.equal(inspectAppliedSkill(variant).applied,false);
});

test("positive autonomous learning requires verified result and semantic validation",async()=>{
 const {classifyLearningOutcome}=await import("../core/learning/learningOutcomePolicy.js");
 const {analyzeExecutionTrace}=await import("../core/learning/experienceAnalyzer.js");
 const good={task:"Date difference",status:"COMPLETED",statistics:{completed:1},
   result:{success:true,verified:true,status:"COMPLETED",
      validation:{valid:true,degraded:false,outcomeType:"result"}},
   experienceUsage:{used:true,applied:true,skills:[{id:"date-difference-calculation"}]}
 };
 assert.equal(classifyLearningOutcome(good).positive,true);
 for(const bad of [
   {...good,result:{...good.result,verified:false}},
   {...good,result:{...good.result,validation:null}},
   {...good,result:{...good.result,validation:{valid:false}}},
   {...good,result:{...good.result,validation:{valid:true,degraded:true}}},
   {...good,result:{...good.result,validation:{valid:true,outcomeType:"no_verified_result"}}}
 ]){
    assert.equal(classifyLearningOutcome(bad).positive,false);
    assert.equal(analyzeExecutionTrace(bad).action,"IGNORE");
 }
});


test("successful selection without real Skill application never reinforces or re-discovers it",async()=>{
 const {analyzeExecutionTrace}=await import("../core/learning/experienceAnalyzer.js");
 const {analyzeExecutionTraceBatch}=await import("../core/learning/multiSkill/multiSkillAnalyzer.js");
 const known={id:"date-difference-calculation",name:"Date Difference",
   workflow:["Use native date_diff"],examples:[]};
 const verified={
   task:"Calculate 9 calendar days",status:"COMPLETED",
   statistics:{completed:1},
   result:{success:true,verified:true,status:"COMPLETED",
     validation:{valid:true,degraded:false,outcomeType:"result"}},
   experienceUsage:{used:true,found:true,skills:[known],
     applied:false,appliedSkillIds:[]}
 };
 const single=analyzeExecutionTrace(verified);
 assert.equal(single.action,"IGNORE");
 assert.equal(single.analysisType,"UNPROVEN_SKILL_APPLICATION");
 assert.equal(single.metadata.positiveReinforcementAllowed,false);
 assert.equal(analyzeExecutionTraceBatch(verified)[0].action,"IGNORE");

 const proven={...verified,experienceUsage:{
   ...verified.experienceUsage,applied:true,appliedSkillIds:[known.id]
 }};
 const improvement=analyzeExecutionTrace(proven);
 assert.notEqual(improvement.analysisType,"UNPROVEN_SKILL_APPLICATION");
 assert.notEqual(improvement.action,"PATTERN_DISCOVERY");
 const noSemanticProof={...proven,result:{...proven.result,verified:false}};
 assert.equal(analyzeExecutionTrace(noSemanticProof).action,"IGNORE");
});

test("complex traces only attribute confirmed application to the exact child Skill",async()=>{
 const {updateTraceFromSummary}=await import("../core/trace/traceResult.js");
 const {createExecutionTrace}=await import("../core/trace/traceFactory.js");
 const {resultUsesSkill}=await import("../core/learning/multiSkill/skillTraceResult.js");
 const one={id:"verified-a"},two={id:"selected-b"};
 const child={success:true,status:"COMPLETED",
  experience:{used:true,found:true,applied:true,appliedSkillIds:[one.id],skills:[one,two]},
  executionMeta:{usedTools:["date_diff"]}
 };
 assert.equal(resultUsesSkill(child,one.id),true);
 assert.equal(resultUsesSkill(child,two.id),false);
 const trace=createExecutionTrace("Calculate interval");
 updateTraceFromSummary(trace,{success:true,status:"COMPLETED",results:[child]});
 assert.equal(trace.experienceUsage.applied,true);
 assert.deepEqual(trace.experienceUsage.appliedSkillIds,[one.id]);
});
