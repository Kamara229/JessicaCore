import test from "node:test";
import assert from "node:assert/strict";
import {parseSequencePriceTask} from "../core/native/sequencePriceIntent.js";
import {calculateSequentialPrice} from "../core/native/sequencePriceCalculation.js";
import {evaluateSequencePrice} from "../core/native/sequencePriceTool.js";
import {tryNativeDecomposition} from "../core/decomposer/nativeDecomposer.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
const task="Цена товара 1000 ₽. Её повысили на 20%, затем снизили на 20%. Какова итоговая цена и процент изменения относительно исходной? Покажи расчёт";
test("whole price question parsed, not split into subrequests",()=>{
 assert.deepEqual(parseSequencePriceTask(task),{price:"1000",increase:20,decrease:20});
 const d=tryNativeDecomposition({task});
 assert.equal(d.handled,true);assert.equal(d.decomposition.isComplex,false);
 assert.equal(d.decomposition.subtasks.length,1);
});
test("sequential percentage calculation is not additive",()=>{
 const r=calculateSequentialPrice(parseSequencePriceTask(task));
 assert.equal(r.afterRaise,1200);assert.equal(r.final,960);assert.equal(r.changePercent,-4);
 assert.equal(calculateSequentialPrice({price:1000,increase:150,decrease:20}),null);
 assert.equal(parseSequencePriceTask("Цена товара 1000 ₽, её повысили на 20%."),null);
});
test("full plan, native answer and verification without model",async()=>{
 const route=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(route.handled,true,JSON.stringify(route.validation));
 const plan=route.plan,step=plan.steps[0];
 assert.equal(step.tool,"sequential_price_percent");
 const result=evaluateSequencePrice(step.arguments);
 const run={success:true,results:[{id:step.id,tool:step.tool,...result}]};
 const answer=await composeAnswer(task,plan,run);
 assert.equal(answer.success,true);assert.equal(answer.source,"tool");
 assert.match(answer.text,/960/);assert.match(answer.text,/-4/);
 const check=await validateResult(task,plan,run,answer);
 assert.equal(check.valid,true);assert.equal(check.degraded,false);
});
