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
});
