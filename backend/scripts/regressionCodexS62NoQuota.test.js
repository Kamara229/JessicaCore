import test from "node:test";
import assert from "node:assert/strict";
import {calibriLaterResearchNoJessicaTest} from "../core/codex/relayCalibriLaterResearchNoTest.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const marker="4aee81dd-e8f3-4b4a-8d35-d212b844d77c";
const body="S62 ещё не завершён. Следующий шаг выполняю через GitHub: проверить native C на восстановленных весах S61, сравнить ответы и скорость CPU с S56. Результаты сохранить в research-ветке. Jessica не тестировать.";
const raw=`JESSICA_REPLY_${marker}\n\n${body}\n\nJESSICA_END_${marker}`;

test("S62 no-test native C message uses deterministic quota-free handoff",async()=>{
  let modelCalls=0;
  const decision=await interpretRelayTurn({message:raw,plan:"CALIBRI 2.0 Green Tunnel"},async()=>{
    modelCalls++;
    throw Object.assign(new Error("429 TPD quota exhausted"),{status:429});
  });
  assert.equal(modelCalls,0,"No Groq requests for explicitly delegated CPU research");
  assert.equal(decision.action,"reply");
  assert.equal(decision.task,"");
  assert.equal(decision.continueThread,false);
  assert.match(decision.reply,/S62/);
  assert.match(decision.reply,/GitHub/);
  assert.match(decision.reply,/Тест в обычном чате не запрашивался/);
});

test("S62 invalid markers and explicit Jessica test are not swallowed",()=>{
  assert.equal(calibriLaterResearchNoJessicaTest(raw.replace(
    "JESSICA_END_"+marker,"JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(raw.replace(
    "Jessica не тестировать.","Выполни тест в обычном чате Jessica.")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(raw.replace(
    "Jessica не тестировать.","Jessica не тестировать. Сделай deploy на Render.")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(raw.replace(
    "Следующий шаг выполняю через GitHub","Следующий шаг выполняю через Render")),null);
});

test("Old CALIBRI S29+ passive no-test wording is preserved",()=>{
 const old="S56: Green Tunnel исследование. Следующий шаг — проверить bpp. Jessica тесты не запускает.";
 const d=calibriLaterResearchNoJessicaTest(old);
 assert.equal(d?.action,"reply");
 assert.equal(d?.task,"");
});
