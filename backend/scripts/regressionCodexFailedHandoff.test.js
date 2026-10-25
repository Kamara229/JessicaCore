import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {acknowledgeBackendProgress} from "../core/codex/relayProgressPolicy.js";

const msg=[
  "JESSICA_REPLY_766118ab-87f4-4052-abdc-0078eff08135",
  "Тест не пройден: селектор не выбрал первоисточник. Исправил backend: Apache 2.0 загружается напрямую с официального URL, без поиска (d09f42c). Следующий шаг — проверить CI и развёртывание Render; тест пока не повторять.",
  "JESSICA_END_766118ab-87f4-4052-abdc-0078eff08135"
].join("\n");

test("failed test followed by a CI/Render step never requires AI interpretation",async()=>{
 let modelCalls=0;
 const decision=await interpretRelayTurn({message:msg,history:[]},async()=>{
   modelCalls++;
   throw Error("provider-unavailable");
 });
 assert.equal(modelCalls,0);
 assert.equal(decision.action,"reply");
 assert.equal(decision.task,"");
 assert.match(decision.reply,/GitHub, Render и Supabase/);
 assert.match(decision.reply,/Тест в обычном чате Jessica пока не запрошен/);
});
test("failure text alone cannot be mistaken for a safe next step",()=>{
 assert.equal(acknowledgeBackendProgress("Тест не пройден: проверь Apache."),null);
 assert.equal(acknowledgeBackendProgress("Тест не пройден. Следующий шаг — введи пароль аккаунта."),null);
 assert.equal(acknowledgeBackendProgress("Тест не пройден. Следующий шаг — введи API ключ для GitHub."),null);
});

test("failed Apache fetch handoff remains deterministic during Groq quota failure",async()=>{
 const message=[
  "JESSICA_REPLY_04df5750-6a63-412a-9bd9-721db63b5b62",
  "Тест не пройден: вместо цитаты §3 выданы строки заголовка. Усилил проверку ответов (eebd1a8). Следующий шаг — подключить прямую загрузку оригинала Apache без TinyFish. Тест пока не повторять.",
  "JESSICA_END_04df5750-6a63-412a-9bd9-721db63b5b62"
 ].join("\n");
 let calls=0;
 const result=await interpretRelayTurn({message,history:[]},async()=>{
   calls++;
   throw Error("no model quota");
 });
 assert.equal(calls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.match(result.reply,/Тест в обычном чате Jessica пока не запрошен/);
});

test("partially passed test hands off Validator and CI without AI retries",async()=>{
 const input=[
  "JESSICA_REPLY_938804dc-f116-4257-8a85-81b89afbe24b",
  "Тест частично пройден: 4 дня — верно, но §3 не указан. Добавил извлечение пункта из первоисточника (5a1b98e).",
  "Следующий шаг — завершить проверку в Validator и CI. Тест пока не повторять.",
  "JESSICA_END_938804dc-f116-4257-8a85-81b89afbe24b"
 ].join("\n");
 let calls=0;
 const decision=await interpretRelayTurn({message:input,history:[]},async()=>{
  calls++;
  throw Error("model-unavailable");
 });
 assert.equal(calls,0);
 assert.equal(decision.action,"reply");
 assert.equal(decision.task,"");
 assert.match(decision.reply,/тест не пройден/);
 assert.match(decision.reply,/Тест в обычном чате Jessica пока не запрошен/);
});

test("Render Live status with pending CI stays in chosen chat without Groq",async()=>{
 const actual=[
  "JESSICA_REPLY_426a3cd9-88e1-4ed1-a619-5ae723cf67c1",
  "Render Live: 78a22a6. Регрессионные тесты прошли, но CI ещё выполняется. Следующий шаг — проверить окончательный статус обоих процессов CI. Тест Jessica пока не повторять.",
  "JESSICA_END_426a3cd9-88e1-4ed1-a619-5ae723cf67c1"
 ].join("\n");
 let calls=0;
 const decision=await interpretRelayTurn({message:actual,history:[]},async()=>{
  calls++;
  throw Error("429 provider quota");
 });
 assert.equal(calls,0);
 assert.equal(decision.action,"reply");
 assert.equal(decision.task,"");
 assert.equal(decision.continueThread,false);
 assert.match(decision.reply,/Проверь указанный шаг через свои подключения/);
 assert.match(decision.reply,/Тест в обычном чате Jessica пока не запрошен/);
 assert.equal(acknowledgeBackendProgress("Render Live: 78a22a6. Регрессии прошли."),null);
});

test("GitHub-first CI progress note bypasses unavailable Groq", async()=>{
 const message=[
  "JESSICA_REPLY_48bebae6-b9d5-42b8-bc45-0b5b43323c06",
  "GitHub: сборка и backend-регрессии 8f66787 успешны. Render Live подтверждён.",
  "Следующий шаг — проверить завершение Phone Chat CI: его проверки ещё выполняются. Тест Jessica пока не повторять.",
  "JESSICA_END_48bebae6-b9d5-42b8-bc45-0b5b43323c06"
 ].join("\n");
 let calls=0;
 const result=await interpretRelayTurn({message,history:[]},async()=>{
  calls++;
  throw Error("provider quota");
 });
 assert.equal(calls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.match(result.reply,/GitHub, Render и Supabase/);
 assert.match(result.reply,/Тест в обычном чате Jessica пока не запрошен/);
 assert.equal(acknowledgeBackendProgress("GitHub: сборка успешна."),null);
});
