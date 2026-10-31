import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {numberedBenchmarkTest} from "../core/codex/relayNumberedBenchmarkTest.js";

const task="Задача Монти Холла: 3 двери, за одной автомобиль. Ведущий знает приз, всегда открывает дверь с козой и предлагает сменить выбор. Стоит ли менять дверь? Объясни вероятности.";
const line="Тест 26/60 пройден: расчёт Байеса верен, ответ 8,33%. Тест 27/60. "+task;
const id="8d99ec19-605c-4952-bbe4-6af59993d6f7";

test("inline benchmark after previous result runs without Groq interpretation",async()=>{
    const forms=[
        line,
        ["JESSICA_REPLY_"+id,line,"JESSICA_END_"+id].join("\n")
    ];
    for(const message of forms){
        let requests=0;
        const actual=await interpretRelayTurn({message},async()=>{
            requests++;
            throw Error("provider 429 quota");
        });
        assert.equal(requests,0);
        assert.deepEqual(actual,{action:"test",reply:"",task,continueThread:false});
    }
});

test("inline benchmark parser rejects unsafe or multiple tasks",()=>{
    assert.equal(numberedBenchmarkTest("Тест 26/60 пройден: правильно."),null);
    assert.equal(numberedBenchmarkTest("Например: "+line),null);
    assert.equal(numberedBenchmarkTest(line+" Не запускай тест."),null);
    assert.equal(numberedBenchmarkTest(line+" Тест 28/60. "+task),null);
    assert.equal(numberedBenchmarkTest("Тест 61/60. "+task),null);
});

const retryTask="В Python функция def add(x, items=[]): items.append(x); return items. Что вернут последовательные вызовы add(1), add(2), add(3, [])? Объясни причину и исправь код.";
const retryBody="Тест 32/60 (повтор после завершения деплоя): "+retryTask;
const retryId="71f4fc99-d9fa-4fec-95f8-45333c28a75a";

test("numbered repeat marker bypasses unavailable Groq interpreter",async()=>{
  for(const message of [retryBody,["JESSICA_REPLY_"+retryId,retryBody,"JESSICA_END_"+retryId].join("\n")]){
    let calls=0;
    const decision=await interpretRelayTurn({message},async()=>{
      calls++;
      throw Error("HTTP 429 quota");
    });
    assert.equal(calls,0);
    assert.deepEqual(decision,{action:"test",reply:"",task:retryTask,continueThread:false});
  }
});

test("repeat notes must not smuggle instructions or an invalid benchmark",()=>{
  assert.equal(numberedBenchmarkTest(retryBody)?.task,retryTask);
  assert.equal(numberedBenchmarkTest("Тест 32/60 (пример): "+retryTask),null);
  assert.equal(numberedBenchmarkTest("Тест 32/60 (повтор): "+retryTask+" Не запускай тест."),null);
  assert.equal(numberedBenchmarkTest("Тест 32/60 (повтор): "+retryTask+"\nТест 33/60: "+retryTask),null);
  assert.equal(numberedBenchmarkTest("Тест 62/60 (повтор): "+retryTask),null);
});
