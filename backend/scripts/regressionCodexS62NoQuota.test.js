import test from "node:test";
import assert from "node:assert/strict";
import {calibriLaterResearchNoJessicaTest} from "../core/codex/relayCalibriLaterResearchNoTest.js";
import {codexQuotaIncidentNoTestStatus} from "../core/codex/relayQuotaIncidentNoTestStatus.js";
import {quotaSafeCalibrIResearchProgress} from "../core/codex/relayQuotaSafeCalibriResearchProgress.js";
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

const incidentID="26e73f25-b325-4a3e-9f9e-5b4122e6bb89";
const incidentBody="Render: Groq 429 — суточный лимит 200 тыс. токенов; Supabase исправен. В research PR #449 проверен обход AI для S62 без теста Jessica. LIVE не менял; лимит провайдера не снят. Далее — native C в GitHub.";
const incident=`JESSICA_REPLY_${incidentID}\n\n${incidentBody}\n\nJESSICA_END_${incidentID}`;

test("verified Render 429 / S62 research relay does not spend Groq quota",async()=>{
  let calls=0;
  const result=await interpretRelayTurn({message:incident,plan:"CALIBRI S62 research"},async()=>{
    calls++;throw new Error("429 quota error");
  });
  assert.equal(calls,0);
  assert.equal(result.action,"reply");
  assert.equal(result.task,"");
  assert.equal(result.continueThread,false);
  assert.match(result.reply,/S62/);
  assert.match(result.reply,/тест Jessica не запускался/);
});

test("429 handoff fails closed on malformed envelopes / new commands",()=>{
  assert.equal(codexQuotaIncidentNoTestStatus(incident.replace(
    "JESSICA_END_"+incidentID,"JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
  assert.equal(codexQuotaIncidentNoTestStatus(incident.replace(
    "Далее — native C в GitHub.","Далее — native C в GitHub. Запусти тест.")),null);
  assert.equal(codexQuotaIncidentNoTestStatus(incident.replace(
    "Далее — native C в GitHub.","Далее — native C в GitHub. Измени Render.")),null);
  assert.equal(codexQuotaIncidentNoTestStatus(incident.replace(
    "S62 без теста Jessica","S62 выполни тест Jessica")),null);
  assert.equal(codexQuotaIncidentNoTestStatus(incident.replace(
    "LIVE не менял","LIVE изменил")),null);
});

const id63="434966b1-2720-4be3-9699-fb26a73248ab";
const body63="S62: восстановление весов native C подтверждено (PR #450). Следующий шаг — S63: ChatGPT сравнит скорость генерации и точность ответов S56/S61 на CPU. Работа только в GitHub research. Jessica не тестировать.";
const envelope63=`JESSICA_REPLY_${id63}\n\n${body63}\n\nJESSICA_END_${id63}`;
test("S63 status avoids Groq 429 and remains a passive unexecuted Jessica test",async()=>{
  let calls=0;
  const response=await interpretRelayTurn({message:envelope63},async()=>{
    calls++;throw Object.assign(new Error("429 daily quota"),{status:429});
  });
  assert.equal(calls,0,"Passively delegated S63 requires no AI provider calls");
  assert.equal(response.action,"reply");
  assert.equal(response.task,"");
  assert.equal(response.continueThread,false);
  assert.match(response.reply,/S62/);
  assert.match(response.reply,/не запрашивался и не запускался/);
});
test("S63 passive status must keep exact delegation, no-action and markers",()=>{
  assert.equal(calibriLaterResearchNoJessicaTest(envelope63.replace("Работа только в GitHub research.","Работа в Jessica.")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(envelope63.replace("Jessica не тестировать.","Jessica, выполни тест.")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(envelope63.replace("JESSICA_END_"+id63,"JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
  assert.equal(calibriLaterResearchNoJessicaTest(envelope63.replace("Jessica не тестировать.","Jessica не тестировать. Сделай deploy.")),null);
});

const id64="72347d33-096e-4cc9-aa30-e691fffc0fa2";
const body64="S64 завершён: Green Tunnel rank128, 0,61337 bpp, ответы 16/16 совпали с S56, скорость CPU 1,005×. Следующий шаг — завершить обучение S65 в GitHub PR #453 и проверить сохранение навыков. Jessica не тестировать.";
const message64=`JESSICA_REPLY_${id64}\n\n${body64}\n\nJESSICA_END_${id64}`;
test("S64-to-S65 GitHub progress bypasses exhausted Groq without Jessica test",async()=>{
 let providerCalls=0;
 const result=await interpretRelayTurn({message:message64},async()=>{
    providerCalls++;throw Object.assign(new Error("429 TPD quota"),{status:429});
 });
 assert.equal(providerCalls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.equal(result.continueThread,false);
 assert.match(result.reply,/S64/);
 assert.match(result.reply,/не запрашивался и не запускался/);
});
test("S64-to-S65 provider-free path refuses malformed, implicit, or executable tasks",()=>{
 assert.equal(calibriLaterResearchNoJessicaTest(message64.replace("JESSICA_END_"+id64,"JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
 assert.equal(calibriLaterResearchNoJessicaTest(message64.replace("Jessica не тестировать.","Jessica, выполни тест.")),null);
 assert.equal(calibriLaterResearchNoJessicaTest(message64.replace("в GitHub PR #453","в Render PR #453")),null);
 assert.equal(calibriLaterResearchNoJessicaTest(message64.replace("Jessica не тестировать.","Jessica не тестировать. Merge.")),null);
});

const handoffID="b63a7ac3-41e0-4500-bcf7-a6aef6779b04";
const handoffBody="Проверил: Render работает, Supabase исправен. Обход Groq 429 прошёл CI в PR #449, но LIVE не обновлён. Следующий шаг — завершить обучение S65 в GitHub PR #453 и проверить сохранение навыков. Jessica не тестировать.";
const handoff='JESSICA_REPLY_'+handoffID+'\n\n'+handoffBody+'\n\nJESSICA_END_'+handoffID;

test("no-test CALIBRI GitHub status bypasses Groq quota instead of generating repeated failure",async()=>{
  let groq=0;
  const decision=await interpretRelayTurn({message:handoff},async()=>{
    groq++;throw Object.assign(new Error("429 TPD"),{status:429});
  });
  assert.equal(groq,0,"No provider call for unambiguous research status");
  assert.deepEqual([decision.action,decision.task,decision.continueThread],["reply","",false]);
  assert.match(decision.reply,/Тест.*не.*запускался/);
});

test("bounded generic CALIBRI relay rejects bad envelopes or implied actions",()=>{
 const expectNo=(input)=>assert.equal(quotaSafeCalibrIResearchProgress(input),null);
 expectNo(handoff.replace("JESSICA_END_"+handoffID,"JESSICA_END_00000000-0000-0000-0000-000000000000"));
 expectNo(handoff.replace("Jessica не тестировать.","Jessica выполни тест."));
 expectNo(handoff.replace("GitHub PR #453","Render PR #453"));
 expectNo(handoff.replace("Jessica не тестировать.","Jessica не тестировать. Deploy на Render."));
 expectNo(handoff.replace("Jessica не тестировать.","Jessica не тестировать. Используй API-ключ."));
 expectNo("S65 GitHub PR #453. Jessica не тестировать.");
});
