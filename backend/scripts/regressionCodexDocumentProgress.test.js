import test from "node:test";
import assert from "node:assert/strict";
import { interpretRelayTurn } from "../core/codex/relayInterpreter.js";
import { documentAnalysisNextStep } from "../core/codex/relayDocumentProgress.js";

const id="af07ddff-e3cb-49a5-a136-ba0c1dd36db4";
const latest=[
 "JESSICA_REPLY_"+id,
 "Добавил контроль подтверждённых цитат по фрагментам (639b182). Неполный анализ не считается полным. Backend-тесты прошли, CI и Render ещё выполняются. Следующий шаг — пораздельный анализ всего документа. Тест пока не нужен.",
 "JESSICA_END_"+id
].join("\n");

test("the real document handoff bypasses HTTP 429 without running a test",async()=>{
 let calls=0;
 const decision=await interpretRelayTurn({message:latest},async()=>{calls++;throw Error("429 quota exceeded");});
 assert.equal(calls,0);
 assert.equal(decision.action,"reply");
 assert.equal(decision.task,"");
 assert.equal(decision.continueThread,false);
 assert.match(decision.reply,/пораздельный анализ/);
 assert.match(decision.reply,/CI и Render/);
});

test("only valid status handoffs are handled offline",()=>{
 assert.ok(documentAnalysisNextStep(latest));
 assert.equal(documentAnalysisNextStep(latest.replace("JESSICA_END_","JESSICA_STOP_")),null);
 assert.equal(documentAnalysisNextStep(latest.replace("пораздельный анализ всего документа","подтвердить оплату")),null);
 assert.equal(documentAnalysisNextStep(latest.replace("Тест пока не нужен","Запусти тест")),null);
 assert.equal(documentAnalysisNextStep(latest.replace("Тест пока не нужен","В обычном чате отправь: «Тест»")),null);
});

test("an actual explicit test always takes priority",async()=>{
 const task="Сколько дней между 10 и 12 октября 2026 года?";
 const decision=await interpretRelayTurn({message:latest+"\nВ обычном чате отправь: «"+task+"»"},
  async()=>{throw Error("provider down");});
 assert.equal(decision.action,"test");
 assert.equal(decision.task,task);
});
