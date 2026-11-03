import test from "node:test";
import assert from "node:assert/strict";
import {onePlainJessicaTest} from "../core/codex/relayOnePlainTest.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const id="764dbe12-a23c-45a0-bbdd-2a3c53e3fd2a";
const task="CALIBRI CB88 занимает 11 413 683 байта при 134 515 008 параметрах. Рассчитай бит/параметр по формуле (байты × 8) / параметры. Округли до 6 знаков и проверь лимит ≤0,7.";
const body="Тест: "+task;
const wrapped=["JESSICA_REPLY_"+id,body,"JESSICA_END_"+id].join("\n");

test("exact REAL Jessica bare calculation test bypasses quota-bound interpreter",async()=>{
  for(const message of [body,wrapped]){
    let providerCalls=0;
    const result=await interpretRelayTurn({message,history:[{role:"user",content:"CALIBRI"}]},
      async()=>{ providerCalls++;throw Error("429 quota exhausted");});
    assert.equal(providerCalls,0);
    assert.deepEqual(result,{action:"test",reply:"",task,continueThread:false});
    assert.deepEqual(onePlainJessicaTest(message),result);
  }
});

test("only one well-formed bounded original test is eligible",()=>{
  for(const message of [
    "JESSICA_REPLY_"+id+"\n"+body+"\nJESSICA_END_00000000-0000-0000-0000-000000000000",
    "JESSICA_REPLY_"+id+"\n"+body,
    body+"\nJESSICA_END_"+id,
    "Тест: "+task+"\nТест: Рассчитай другой показатель.",
    "Например, Тест: "+task,
    "Не запускай. "+body,
    "Тест не запускался. "+body,
    "Тест: "+task+" Повтори тест: Вычисли хеш.",
    "Тест: CALIBRI CB88: результаты не измерены.",
    "Шаг: рассчитай количество бит в архиве.",
    "Проверил 11 413 683 байта. Тест пока не запускать."
  ])assert.equal(onePlainJessicaTest(message),null,message);
});

test("all non-tests stay fail-closed even when Groq is exhausted",async()=>{
 const message="Кодекс Jessica: анализ последнего сообщения ChatGPT не выполнен — сервис интерпретации вернул ошибку. Тест не запускался.";
 let calls=0;
 const r=await interpretRelayTurn({message},async()=>{calls++;throw Error("quota unavailable");});
 assert.equal(r.action,"reply");
 assert.equal(r.task,"");
 assert.ok(calls<=1);
 assert.equal(onePlainJessicaTest(message),null);
});

test("no test execution or repair/learning side effects are authorized",()=>{
 const result=onePlainJessicaTest(wrapped);
 assert.equal(result.action,"test");
 assert.equal(result.continueThread,false);
 assert.equal(result.task,task);
 assert.ok(!("execute" in result));
 assert.ok(!("testSuccess" in result));
});
