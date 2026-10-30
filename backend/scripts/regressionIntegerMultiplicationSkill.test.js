import test from "node:test";
import assert from "node:assert/strict";
import {parseIntegerProductTask} from "../core/native/integerMultiplicationIntent.js";
import {searchExperience} from "../experience/search/experienceSearch.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {multiplyIntegers} from "../tools/integerMultiplicationTool.js";

const skill={id:"integer-multiplication",name:"Integer Multiplication",enabled:true,status:"published",confidence:.8};
const task="Вычисли 247 × 306. Покажи расчёт";
test("published multiplication Skill is selected for one exact integer product",()=>{
  const found=searchExperience(task,[skill]);
  assert.equal(found.found,true);
  assert.equal(found.experience.id,skill.id);
  const result=tryCreateNativeSkillPlan({task,executionExperience:{skills:[skill]},planningContext:{}});
  assert.equal(result.handled,true);
  assert.equal(result.plan.steps[0].tool,"integer_multiply");
  assert.equal(result.plan.experience.used,true);
});
test("exact multiplication with explanations and negative numbers",()=>{
  const r=multiplyIntegers({a:"-247",b:"306",showWork:true});
  assert.equal(r.data.product,"-75582");
  assert.match(r.text,/-74100/);
  assert.equal(multiplyIntegers({a:"247",b:"306"}).data.product,"75582");
});
test("do not hijack complex or ambiguous calculations and never attribute disabled Skills",()=>{
  assert.equal(parseIntegerProductTask("Вычисли 247 × 306 × 2"),null);
  assert.equal(parseIntegerProductTask("Вычисли 247 × 306 + 1"),null);
  assert.equal(searchExperience(task,[{...skill,enabled:false}]).found,false);
  const fallback=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]},planningContext:{}});
  assert.equal(fallback.handled,true);
  assert.equal(fallback.plan.experience.used,false);
});
test("production decomposer wording keeps the native Skill route under AI 429",()=>{
  const decomposed="Вычислить произведение 247 × 306 и показать пошаговый расчёт умножения.";
  const parsed=parseIntegerProductTask(decomposed);
  assert.deepEqual(parsed,{a:"247",b:"306",showWork:true});
  const experience=searchExperience(decomposed,[skill]);
  assert.equal(experience.found,true);
  assert.equal(experience.experience.id,skill.id);
  const planned=tryCreateNativeSkillPlan({
    task:decomposed,executionExperience:{skills:[experience.experience]},planningContext:{}
  });
  assert.equal(planned.handled,true);
  assert.equal(planned.plan.steps[0].tool,"integer_multiply");
  assert.equal(planned.plan.steps[0].arguments.showWork,true);
  assert.equal(planned.plan.experience.used,true);
  assert.equal(multiplyIntegers(planned.plan.steps[0].arguments).data.product,"75582");
});
test("natural explanation text works without allowing extra operations",()=>{
  assert.deepEqual(parseIntegerProductTask("Вычислить произведение (-247) × 306 и показать расчёт"),
    {a:"-247",b:"306",showWork:true});
  assert.equal(parseIntegerProductTask("Calculate 247 x 306, explain the steps")?.showWork,true);
  assert.equal(parseIntegerProductTask("Вычислить произведение 247 × 306 и 100 × 2"),null);
  assert.equal(parseIntegerProductTask("Вычислить произведение 247 × 306 + 10"),null);
});

test("real task runner executes decomposed product without external model",async()=>{
  const {buildExecutionExperienceContext}=await import("../core/subtask/subtaskExperienceContext.js");
  const {createExecutionContext}=await import("../core/execution/context/contextFactory.js");
  const {executeTaskRunner}=await import("../core/execution/stepExecution/taskRunnerExecutor.js");
  const decomposed="Вычислить произведение 247 × 306 и показать пошаговый расчёт умножения.";
  const chosen=buildExecutionExperienceContext({
    found:true,source:"experience-search",confidence:.7,experience:skill,
    planningContext:{experience:{id:skill.id}}
  });
  const native=tryCreateNativeSkillPlan({
    task:decomposed,executionExperience:chosen,planningContext:{}
  });
  assert.equal(native.handled,true);
  const context=createExecutionContext({
    task:decomposed,plan:native.plan,planningContext:{},experience:chosen
  });
  const result=await executeTaskRunner(context);
  assert.equal(result.success,true,JSON.stringify(result.failure||{}));
  assert.equal(context.runResult?.results?.[0]?.data?.product,"75582");
  assert.equal(context.experience.used,true);
});

test("exact native answer receives proof-based validation without an AI provider",async()=>{
  const {buildExecutionExperienceContext}=await import("../core/subtask/subtaskExperienceContext.js");
  const {createExecutionContext}=await import("../core/execution/context/contextFactory.js");
  const {executeTaskRunner}=await import("../core/execution/stepExecution/taskRunnerExecutor.js");
  const {validateResult}=await import("../core/validator.js");
  const {verifiedNativeIntegerProduct}=await import("../core/validator/nativeIntegerProductProof.js");
  const input="Вычислить произведение 247 × 306 и показать пошаговый расчёт умножения.";
  const chosen=buildExecutionExperienceContext({
    found:true,source:"experience-search",confidence:.7,
    experience:skill,planningContext:{experience:{id:skill.id}}
  });
  const route=tryCreateNativeSkillPlan({
    task:input,executionExperience:chosen,planningContext:{}
  });
  assert.equal(route.handled,true);
  const context=createExecutionContext({
    task:input,plan:route.plan,planningContext:{},experience:chosen
  });
  assert.equal((await executeTaskRunner(context)).success,true);
  const text=context.runResult.results[0].text;
  const reply={success:true,source:"native-fallback",text};
  assert.equal(verifiedNativeIntegerProduct(input,route.plan,context.runResult,reply),true);
  const validation=await validateResult(input,route.plan,context.runResult,reply);
  assert.equal(validation.valid,true);
  assert.equal(validation.validationMode,"tool_direct");
  assert.equal(validation.degraded,false);
  assert.equal(verifiedNativeIntegerProduct(input,route.plan,context.runResult,{
    ...reply,text:text+" чужое утверждение"
  }),false);
  assert.equal(verifiedNativeIntegerProduct(input,route.plan,context.runResult,{
    ...reply,source:"web-fetch"
  }),false);
  assert.equal(verifiedNativeIntegerProduct(input,route.plan,{
    ...context.runResult,results:[{...context.runResult.results[0],data:{product:"3"}}]
  },reply),false);
  assert.equal(verifiedNativeIntegerProduct(input,route.plan,{
    ...context.runResult,results:[{...context.runResult.results[0],success:false}]
  },reply),false);
});

test("native product Composer skips AI and remains strictly proof-based",async()=>{
 const {composeAnswer}=await import("../core/answerComposer.js");
 const {validateResult}=await import("../core/validator.js");
 const task="Вычисли (-247) × 306. Покажи расчёт";
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[skill]},planningContext:{}});
 const step=planned.plan.steps[0];
 const result=multiplyIntegers(step.arguments);
 const run={success:true,results:[{...result,id:step.id,tool:step.tool}]};
 const answer=await composeAnswer(task,planned.plan,run);
 assert.equal(answer.source,"native-integer-proof");
 assert.equal(answer.success,true);
 assert.equal((await validateResult(task,planned.plan,run,answer)).validationMode,"tool_direct");
 const proof=(await import("../core/composer/verifiedIntegerProductAnswer.js")).composeVerifiedIntegerProduct;
 assert.equal(proof(task,planned.plan,{...run,results:[{...run.results[0],text:"incorrect"}]}),null);
});
test("native decomposer preserves signed product without calling AI",async()=>{
 const {tryNativeDecomposition}=await import("../core/decomposer/nativeDecomposer.js");
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const tasks=["Вычисли 247 × 306. Покажи расчёт",
   "Вычисли (-247) × (-306). Покажи расчёт"];
 for(const task of tasks){
   const raw=tryNativeDecomposition({task,memoryContext:{skills:[]}});
   assert.equal(raw.handled,true);
   assert.equal(raw.source,"native-integer-product");
   assert.deepEqual(raw.decomposition.subtasks.map(s=>s.text),[task]);
   const result=await decomposeTask({task,memoryContext:{skills:[]}});
   assert.equal(result.success,true);
   assert.equal(result.native,true);
   assert.equal(result.source,"native-integer-product");
   assert.equal(result.decomposition.subtasks[0].text,task);
 }
});
test("ambiguous arithmetic stays out of single-product native decomposition",async()=>{
 const {tryNativeDecomposition}=await import("../core/decomposer/nativeDecomposer.js");
 for(const task of ["Вычисли 247 × 306 + 1","Вычисли 247 × 306 × 2",
   "Сравни 247 × 306 с 250 × 300"]){
    const parsed=tryNativeDecomposition({task,memoryContext:{skills:[]}});
    assert.notEqual(parsed.source,"native-integer-product");
 }
});
