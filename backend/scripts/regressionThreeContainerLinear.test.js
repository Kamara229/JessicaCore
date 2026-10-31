import test from "node:test";
import assert from "node:assert/strict";
import {parseThreeContainerLinearTask} from "../core/native/threeContainerLinearIntent.js";
import {evaluateThreeContainerLinear} from "../core/native/threeContainerLinearTool.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";

const task="В трёх корзинах целое число яблок. В первой вдвое больше, чем во второй; в третьей на 3 меньше, чем в первой. Всего 12. Сколько в каждой? Проверь";
test("parse full input and skip missing premises",()=>{
 assert.deepEqual(parseThreeContainerLinearTask(task),
  {multiple:2,offset:3,offsetSign:-1,sum:12});
 assert.equal(parseThreeContainerLinearTask("В трёх корзинах яблоки. Всего 12."),null);
});
test("independent integer equations, including no feasible integer solution",()=>{
 for(const [input,counts] of [
  [{multiple:2,offset:3,offsetSign:-1,sum:12},[6,3,3]],
  [{multiple:3,offset:4,offsetSign:1,sum:32},[12,4,16]],
  [{multiple:2,offset:2,offsetSign:-1,sum:13},[6,3,4]]
 ]){
  const r=evaluateThreeContainerLinear(input);
  assert.equal(r.success,true);
  assert.deepEqual([r.data.first,r.data.second,r.data.third],counts);
 }
 const impossible=evaluateThreeContainerLinear({multiple:2,offset:3,offsetSign:-1,sum:11});
 assert.equal(impossible.data.feasible,false);
});
test("planner routes to verified native calculation without a web search",async()=>{
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planned.handled,true,JSON.stringify(planned.validation));
 const plan=planned.plan,step=plan.steps[0];
 assert.equal(step.tool,"three_container_linear");
 const result=evaluateThreeContainerLinear(step.arguments);
 const run={success:true,results:[{id:step.id,tool:step.tool,...result}]};
 const answer=await composeAnswer(task,plan,run);
 assert.equal(answer.success,true);
 assert.equal(answer.text,result.text);
 assert.match(answer.text,/первая — 6, вторая — 3, третья — 3/);
 const check=await validateResult(task,plan,run,answer);
 assert.equal(check.valid,true);assert.equal(check.degraded,false);
});
