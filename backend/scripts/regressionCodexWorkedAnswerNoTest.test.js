import test from "node:test";
import assert from "node:assert/strict";
import {workedAnswerNoTest} from "../core/codex/relayWorkedAnswerNoTest.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const id="4f75333d-ac5d-4f0a-ab43-d06849bff141";
const worked=[
  "JESSICA_REPLY_"+id,
  "247 × 306 = 247 × (300 + 6)",
  "247 × 300 = 74 100",
  "247 × 6 = 1 482",
  "74 100 + 1 482 = **75 582**",
  "JESSICA_END_"+id
].join("\n");

test("actual solved answer is a no-test handoff without model calls",async()=>{
  let calls=0;
  const result=await interpretRelayTurn({message:worked},async()=>{
    calls++;throw Error("provider unavailable");
  });
  assert.equal(calls,0);
  assert.equal(result.action,"reply");
  assert.equal(result.task,"");
  assert.equal(result.continueThread,false);
  assert.match(result.reply,/Тест не запускался/u);
});
test("explicit ordinary-chat test still takes priority",async()=>{
  const task="Вычисли 247 × 306. Покажи расчёт";
  const message="В обычном чате Jessica отправь: «"+task+"». Передай ответ дословно.";
  let calls=0;
  const result=await interpretRelayTurn({message},async()=>{
    calls++;throw Error("provider unavailable");
  });
  assert.equal(calls,0);
  assert.equal(result.action,"test");
  assert.equal(result.task,task);
});
test("no-test handler cannot accept changed markers or prose",()=>{
  assert.equal(workedAnswerNoTest(worked.replace("JESSICA_END_"+id,"JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
  assert.equal(workedAnswerNoTest("В обычном чате Jessica отправь: «Вычисли 247 × 306»"),null);
});