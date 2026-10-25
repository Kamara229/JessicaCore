import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {compactJessicaTest} from "../core/codex/relayCompactTest.js";

test("actual compact Codex test bypasses unavailable interpretation provider",async()=>{
 const task="Сколько дней с 27 февраля по 2 марта 2024 года (27-е исключить, 2-е включить)? Объясни расчёт";
 const message=[
  "JESSICA_REPLY_549c3b2c-a1e7-43a2-a34c-82e4a4d7306d",
  "Исправлен сдвиг даты (bdbc211; регрессионный тест CI пройден). В обычном чате отправь: «"+task+"». Пришли ответ дословно.",
  "JESSICA_END_549c3b2c-a1e7-43a2-a34c-82e4a4d7306d"
 ].join("\n");
 const result=await interpretRelayTurn({message},async()=>{throw Error("model unavailable")});
 assert.equal(result.action,"test");
 assert.equal(result.task,task);
 assert.equal(result.continueThread,false);
});

test("compact parser respects plain commands and rejects examples or negated/ambiguous instructions",()=>{
 assert.equal(compactJessicaTest("Например: «Сколько будет 2+2?»"),null);
 assert.equal(compactJessicaTest("Не отправляй. В обычном чате отправь: «Сколько будет 2+2?»"),null);
 assert.equal(compactJessicaTest("В обычном чате отправь: «Первый тест».\nТест в обычном чате: «Второй тест»"),null);
 assert.equal(compactJessicaTest("В обычном чате отправь: «Сколько будет 2+2?»")?.task,"Сколько будет 2+2?");
});
