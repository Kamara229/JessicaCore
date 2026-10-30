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
