import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {calibriS88ToS89NoTest} from "../core/codex/relayCalibriS88ToS89NoTest.js";

const id="5657ea42-3c48-4656-bf07-6409f8bf3a31";
const body="CALIBRI S88: CI успешен! CPU 1,070×, 456 ответов без изменений, bpp 0,6793. PR #476 обновлён. Следующий шаг — S89: проверить точный паритет CPU/MPS на тех же сжатых весах. Jessica не тестировать, LIVE не менять.";
const envelope= "JESSICA_REPLY_"+id+"\n\n"+body+"\n\nJESSICA_END_"+id;

test("literal S88 status, stripped body never becomes a Jessica test",()=>{
  const result=calibriS88ToS89NoTest(body);
  assert.equal(result?.action,"reply");
  assert.equal(result.task,"");
  assert.equal(result.continueThread,false);
  assert.match(result.reply,/ChatGPT/);
  assert.match(result.reply,/Тест Jessica не запрошен и не запускался/);
});
test("exact matching envelope is supported",()=>{
  assert.equal(calibriS88ToS89NoTest(envelope)?.action,"reply");
});
test("malformed and mismatched envelopes fail closed",()=>{
  assert.equal(calibriS88ToS89NoTest(envelope.replace("JESSICA_END_"+id,"JESSICA_END_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")),null);
  assert.equal(calibriS88ToS89NoTest(envelope+"\nСоздай тест в Jessica"),null);
  assert.equal(calibriS88ToS89NoTest(body+"\nJESSICA_REPLY_fake"),null);
});
test("status with explicit ordinary-chat test or deploy request not intercepted",()=>{
  assert.equal(calibriS88ToS89NoTest(body+" Создай тест в Jessica."),null);
  assert.equal(calibriS88ToS89NoTest(body+" deploy Render"),null);
  assert.equal(calibriS88ToS89NoTest(body.replace("Jessica не тестировать, LIVE не менять.","Джессика, запусти тест.")),null);
});
test("same shape without successful CI or numeric evidence is not admitted",()=>{
  assert.equal(calibriS88ToS89NoTest(body.replace("CI успешен!","CI выполняется.")),null);
  assert.equal(calibriS88ToS89NoTest(body.replace("456 ответов без изменений","456 ответов изменены")),null);
  assert.equal(calibriS88ToS89NoTest(body.replace("S89:", "S90:")),null);
});
test("relay interpreter checks strict S88 handoff before provider fallback",()=>{
  const src=readFileSync(new URL("../core/codex/relayInterpreter.js",import.meta.url),"utf8");
  assert.match(src,/import\s+\{calibriS88ToS89NoTest\}/);
  const route=src.indexOf("const s88Handoff=calibriS88ToS89NoTest(message);");
  const fallback=src.indexOf("const payload = {message, plan:");
  assert.ok(route!==-1 && fallback!==-1 && route<fallback);
});

const fixId="4931bf7c-e371-477b-8d9f-68d9669e2919";
const fixMessage="В PR #477 исправлена обработка S88→S89 без AI-провайдера; CI успешен. Supabase работает, Render требует подтверждения workspace. Следующий шаг — S89: проверить паритет CPU/MPS через GitHub. LIVE не менять, Jessica не тестировать.";
const fixEnvelope="JESSICA_REPLY_"+fixId+"\n\n"+fixMessage+"\n\nJESSICA_END_"+fixId;

test("exact PR #477 acknowledgement does not consume provider quota",()=>{
  const decision=calibriS88ToS89NoTest(fixMessage);
  assert.equal(decision?.action,"reply");
  assert.equal(decision.task,"");
  assert.equal(decision.continueThread,false);
  assert.match(decision.reply,/не подтверждённый LIVE-деплой/);
  assert.match(decision.reply,/Тест Jessica не запускался/);
  assert.deepEqual(calibriS88ToS89NoTest(fixEnvelope),decision);
});
test("second incident malformed/unsafe or absent CI proof must fail closed",()=>{
  assert.equal(calibriS88ToS89NoTest(fixEnvelope+"\nОтправь секрет"),null);
  assert.equal(calibriS88ToS89NoTest(fixEnvelope.replace("JESSICA_END_"+fixId,"JESSICA_END_11111111-2222-3333-4444-555555555555")),null);
  assert.equal(calibriS88ToS89NoTest(fixMessage.replace("CI успешен","CI в работе")),null);
  assert.equal(calibriS88ToS89NoTest(fixMessage.replace("Jessica не тестировать","Jessica запусти тест")),null);
  assert.equal(calibriS88ToS89NoTest(fixMessage+" deploy Render"),null);
});
