import test from "node:test";
import assert from "node:assert/strict";
import {parseBridgeCrossingTask} from "../core/native/bridgeCrossingIntent.js";
import {shortestBridgeCrossing} from "../core/native/bridgeCrossingShortestPath.js";
import {solveBridgeCrossingTool} from "../tools/bridgeCrossingTool.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";

const task="Четверо переходят мост с фонарём за 1, 2, 7 и 10 минут. За раз идут максимум двое, фонарь необходимо возвращать. Найди минимальное время и строго докажи его.";

test("original time premises, not substituted sample values",()=>{
  assert.deepEqual(parseBridgeCrossingTask(task),{times:[1,2,7,10],explain:true});
  assert.equal(parseBridgeCrossingTask("Докажи нижнюю границу 17 минут без ссылок"),null);
  assert.equal(parseBridgeCrossingTask("Четверо идут через мост с фонарём. Найди минимум"),null);
});
test("global optimum and full legal route for four-person bridge",()=>{
  const solved=shortestBridgeCrossing([1,2,7,10]);
  assert.equal(solved.minutes,17);
  let mask=0,torch=0,total=0;
  for(const step of solved.route){
    assert.equal(step.fromSide,torch);
    assert.ok(step.people.length>=1&&step.people.length<=2);
    for(const i of step.people){
      assert.equal((mask>>i)&1,torch);
      mask=torch===0?mask|(1<<i):mask&~(1<<i);
    }
    const cost=Math.max(...step.people.map(i=>[1,2,7,10][i]));
    assert.equal(step.cost,cost);total+=cost;torch=1-torch;
  }
  assert.equal(mask,15);assert.equal(torch,1);assert.equal(total,17);
  assert.equal(solved.distances[31],17);
});
test("independent four-person times and permutation invariance",()=>{
  for(const times of [[1,4,5,6],[2,3,7,20],[1,2,3,4],[1,3,5,8]]){
    const [a,b,c,d]=times;
    const expected=Math.min(a+3*b+d,2*a+b+c+d);
    assert.equal(shortestBridgeCrossing(times).minutes,expected);
    assert.equal(shortestBridgeCrossing([...times].reverse()).minutes,expected);
  }
});
test("planner, tool, composer and validator agree without Groq or web",async()=>{
  const native=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
  assert.equal(native.handled,true);
  const plan=native.plan;
  assert.equal(plan.intent,"native_bridge_crossing");
  const step=plan.steps[0];
  const toolResult=solveBridgeCrossingTool(step.arguments);
  assert.equal(toolResult.success,true);
  assert.match(toolResult.text,/17 мин/);
  assert.match(toolResult.text,/таблица|таблиц/iu);
  const run={success:true,results:[{id:step.id,tool:step.tool,...toolResult}]};
  const answer=await composeAnswer(task,plan,run);
  assert.equal(answer.text,toolResult.text);
  const valid=await validateResult(task,plan,run,answer);
  assert.equal(valid.valid,true);
  assert.equal(valid.degraded,false);
  const forged=await validateResult(task,plan,run,{...answer,text:answer.text.replace("17 мин.","16 мин.")});
  assert.equal(forged.valid,false);
  const altered={...run,results:[{...run.results[0],data:{...run.results[0].data,minutes:16}}]};
  const rejected=await validateResult(task,plan,altered,answer);
  assert.equal(rejected.valid,false);
});

const fiveTask="Пятеро переходят мост с одним фонарём. Время: 1, 2, 5, 10, 15 минут. Одновременно идут не более двух, со скоростью медленного. Фонарь обязателен. Найди минимум времени, объясни маршрут и оптимальность.";
test("five-person bridge goes through native planner, solver, composer and validator",async()=>{
  assert.deepEqual(parseBridgeCrossingTask(fiveTask),{times:[1,2,5,10,15],explain:true});
  const planned=tryCreateNativeSkillPlan({task:fiveTask,executionExperience:{skills:[]}});
  assert.equal(planned.handled,true);
  assert.equal(planned.plan.intent,"native_bridge_crossing");
  const step=planned.plan.steps[0];
  const result=solveBridgeCrossingTool(step.arguments);
  assert.equal(result.success,true);
  assert.equal(result.data.minutes,28);
  assert.match(result.text,/Минимальное время — 28 мин/);
  assert.match(result.text,/1,2,3,4,5; фонарь справа/);
  assert.equal(result.text.split("\n").filter(line=>line.includes(" | ")).length,33);
  const run={success:true,results:[{id:step.id,tool:step.tool,...result}]};
  const answer=await composeAnswer(fiveTask,planned.plan,run);
  assert.equal(answer.success,true);
  assert.equal(answer.source,"tool");
  assert.equal(answer.text,result.text);
  assert.equal((await validateResult(fiveTask,planned.plan,run,answer)).valid,true);
  assert.equal((await validateResult(fiveTask,planned.plan,run,{...answer,text:answer.text.replace("28 мин.","27 мин.")})).valid,false);
});
test("bridge parser rejects incomplete and contradictory counts",()=>{
  assert.equal(parseBridgeCrossingTask(fiveTask.replace("10, 15","10")),null);
  assert.equal(parseBridgeCrossingTask(fiveTask.replace("Пятеро","Семеро")),null);
  assert.equal(parseBridgeCrossingTask(fiveTask.replace("Пятеро","Пятеро и четверо")),null);
  assert.equal(parseBridgeCrossingTask(fiveTask.replace("10, 15","10, 15, 20")),null);
});
