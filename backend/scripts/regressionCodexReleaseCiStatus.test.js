import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {releaseCiStatusHandoff} from "../core/codex/relayReleaseCiStatus.js";

const id="742fc3ab-5016-4684-816c-9a6734f0df0c";
const actual=[
  "JESSICA_REPLY_"+id,
  "Добавил пораздельный анализ документов с проверкой цитат и учётом обработанных фрагментов (5b879be). Render Live, backend-тесты прошли. Следующий шаг — проверить завершение общего CI. Тест пока не нужен.",
  "JESSICA_END_"+id
].join("\n");

test("actual document status skips HTTP 429 interpretation without an ordinary Jessica test",async()=>{
  let calls=0;
  const decision=await interpretRelayTurn({message:actual},async()=>{
      calls++;throw Error("429 quota exceeded");
  });
  assert.equal(calls,0);
  assert.equal(decision.action,"reply");
  assert.equal(decision.task,"");
  assert.equal(decision.continueThread,false);
  assert.match(decision.reply,/финальный статус CI/);
  assert.match(decision.reply,/Тест в обычном чате Jessica пока не нужен/);
});

test("safe release progress is not tied to one commit or document name",()=>{
  const another=actual.replace("пораздельный анализ документов","отдельные проверки")
    .replace("(5b879be)","(a123f45)");
  assert.ok(releaseCiStatusHandoff(another));
});

test("unsafe, incomplete and ambiguous handoffs never bypass model reasoning",()=>{
  for(const invalid of [
    actual.replace("JESSICA_END_","JESSICA_STOP_"),
    actual.replace("(5b879be)",""),
    actual.replace("Render Live","Render ещё запускается"),
    actual.replace("Тест пока не нужен","Запусти тест"),
    actual.replace("проверить завершение общего CI","установить APK"),
    actual.replace("проверить завершение общего CI","удалить базу данных"),
    actual.replace("backend-тесты прошли","backend-тесты не прошли")
  ])assert.equal(releaseCiStatusHandoff(invalid),null);
});

test("explicitly requested test outranks a release progress note",async()=>{
  const task="Сколько дней между 10 и 12 октября 2026 года?";
  const message=actual+"\nВ обычном чате отправь: «"+task+"»";
  const decision=await interpretRelayTurn({message},async()=>{throw Error("429 quota");});
  assert.equal(decision.action,"test");
  assert.equal(decision.task,task);
});
