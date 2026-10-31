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
