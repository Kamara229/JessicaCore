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
