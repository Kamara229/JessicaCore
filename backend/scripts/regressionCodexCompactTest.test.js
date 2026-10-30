import "./regressionCodexWorkedAnswerNoTest.test.js";
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


test("next Jessica test on a following line never depends on exhausted analysis provider",async()=>{
 const task="Вычисли количество дней между 29 октября и 6 ноября 2026 года, исключая 29 октября и включая 6 ноября. Объясни расчёт";
 const message=[
  "JESSICA_REPLY_952b1df1-8639-42b1-ae4f-058b41630927",
  "GitHub CI и live-smoke прошли. Render Live: 9780703.",
  "",
  "Следующий тест в обычном чате Jessica:",
  "",
  "«"+task+"».",
  "",
  "Верни ответ дословно.",
  "JESSICA_END_952b1df1-8639-42b1-ae4f-058b41630927"
 ].join("\n");
 let calls=0;
 const decision=await interpretRelayTurn({message},async()=>{calls++;throw Error("429 Groq quota")});
 assert.equal(calls,0);
 assert.equal(decision.action,"test");
 assert.equal(decision.task,task);
 assert.equal(decision.continueThread,false);
});

test("next-test extraction refuses ambiguity and negated commands",()=>{
 const task="Вычисли количество дней между 29 октября и 6 ноября 2026 года";
 assert.equal(compactJessicaTest("Следующий тест в обычном чате Jessica:\n\n«"+task+"»")?.task,task);
 assert.equal(compactJessicaTest("Следующий тест в обычном чате Jessica: «"+task+"»\nТест в обычном чате: «Сколько будет 7 плюс 8?»"),null);
 assert.equal(compactJessicaTest("Не запускай. Следующий тест в обычном чате Jessica: «"+task+"»"),null);
});


test("exact restart test handoffs bypass unavailable interpreter in both word orders",async()=>{
 const task="После рестарта Candidate=PROMOTED, владелец P1/S1, claim сохранён, ACK потерян. Readback недоступен. P2 пытается создать Skill. Что делать до и после восстановления readback?";
 const messages=[
  [
   "JESSICA_REPLY_cfd8a6c8-ad03-4c0a-9bb1-00cdf9fe740a",
   "Отправь в обычный чат Jessica:",
   "",
   "«"+task+"»",
   "",
   "Пришли ответ Jessica дословно.",
   "JESSICA_END_cfd8a6c8-ad03-4c0a-9bb1-00cdf9fe740a"
  ].join("\n"),
  [
   "JESSICA_REPLY_ff021a46-319d-4624-857f-6d77f4dcf325",
   "В обычном чате Jessica отправь: «"+task+"» Пришли ответ дословно.",
   "JESSICA_END_ff021a46-319d-4624-857f-6d77f4dcf325"
  ].join("\n")
 ];
 for(const message of messages){
  let calls=0;
  const decision=await interpretRelayTurn({message,history:[]},async()=>{
   calls++;throw Error("429 exhausted interpreter");
  });
  assert.equal(calls,0,"explicit test must not use interpretation provider");
  assert.deepEqual(decision,{action:"test",reply:"",task,continueThread:false});
 }
});

test("ordinary-chat test extractor rejects negation and multiple quoted tests",()=>{
 const cmd="Отправь в обычный чат Jessica:\n\n«Проверь восстановление после рестарта.»";
 assert.equal(compactJessicaTest("Не отправляй. "+cmd),null);
 assert.equal(compactJessicaTest(cmd+"\nОтправь в обычный чат Jessica: «Выполни другой тест.»"),null);
 assert.equal(compactJessicaTest("Например, «Проверь восстановление после рестарта»."),null);
 assert.equal(compactJessicaTest("Отправь в обычный чат Jessica: «Проверь восстановление после рестарта.»")?.task,
  "Проверь восстановление после рестарта.");
});

test("exact 2026-10-30 failed BSD table + next Jessica test bypasses unavailable model", async()=>{
 const body="Тест не пройден: содержание верное, но формат нарушен. Публикация исправления заблокирована GitHub. Следующий тест в Jessica: «Сравни MIT и BSD 3-Clause. Ответь только таблицей». Передай ответ дословно.";
 const task="Сравни MIT и BSD 3-Clause. Ответь только таблицей";
 const wrapped=["JESSICA_REPLY_66858998-9f68-47bd-a625-de4db084d9fa",body,"JESSICA_END_66858998-9f68-47bd-a625-de4db084d9fa"].join("\n");
 for(const message of [body,wrapped]){
  let calls=0;
  const decision=await interpretRelayTurn({message,history:[]},async()=>{
    calls++; throw Error("429 unavailable");
  });
  assert.equal(calls,0);
  assert.deepEqual(decision,{action:"test",reply:"",task,continueThread:false});
 }
});

test("short next-test variant rejects negation and multiple commands",()=>{
 const testCommand="Следующий тест в Jessica: «Сравни MIT и BSD 3-Clause. Ответь только таблицей»";
 assert.equal(compactJessicaTest("Не запускай. "+testCommand),null);
 assert.equal(compactJessicaTest(testCommand+"\nСледующий тест в Jessica: «Проверь другую лицензию»"),null);
 assert.equal(compactJessicaTest("Пример: «Сравни MIT и BSD 3-Clause»"),null);
});

test("exact GPL section 4 retest directive bypasses unavailable interpretation provider",async()=>{
 const task="Приведи дословную цитату из §4 GPL 3.0 о copyright notice. Укажи официальный URL";
 const message=[
  "JESSICA_REPLY_e6eb76c5-d349-490d-986e-56d64cfcf508",
  "Исправление PR #83 развёрнуто, CI пройден, Render LIVE. В обычном чате Jessica повтори тест: «"+task+"». Передай ответ дословно.",
  "JESSICA_END_e6eb76c5-d349-490d-986e-56d64cfcf508"
 ].join("\n");
 let modelCalls=0;
 const parsed=await interpretRelayTurn({message,history:[]},async()=>{
  modelCalls++; throw Error("429 quota unavailable");
 });
 assert.equal(modelCalls,0);
 assert.deepEqual(parsed,{action:"test",reply:"",task,continueThread:false});
});

test("repeat test directive remains fail-closed for negation and ambiguity",()=>{
 const first="В обычном чате Jessica повтори тест: «Проверь цитату §4 GPLv3.»";
 assert.equal(compactJessicaTest(first)?.task,"Проверь цитату §4 GPLv3.");
 assert.equal(compactJessicaTest("Повтори тест в обычном чате Jessica: «Проверь цитату §4 GPLv3.»")?.task,
  "Проверь цитату §4 GPLv3.");
 assert.equal(compactJessicaTest("Не повторяй. "+first),null);
 assert.equal(compactJessicaTest(first+"\nСледующий тест в Jessica: «Проверь цитату §3 Apache.»"),null);
 assert.equal(compactJessicaTest("Например: «Проверь цитату §4 GPLv3.»"),null);
});

test("failed GPL quote fix PR and explicit no-repeat message bypasses model interpretation",async()=>{
 const message=[
  "JESSICA_REPLY_6675c009-2aec-4b2f-b738-90b2cceb0c28",
  "Сбой подтверждён: декомпозиция меняет исходный запрос, после чего проверка источника отклоняет нужный фрагмент. Исправление подготовлено в PR #85, но CI не пройден. Повторный тест пока не запускай.",
  "JESSICA_END_6675c009-2aec-4b2f-b738-90b2cceb0c28"
 ].join("\n");
 let aiCalls=0;
 const result=await interpretRelayTurn({message,history:[]},async()=>{
  aiCalls++;throw Error("429 AI interpretation unavailable");
 });
 assert.equal(aiCalls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.equal(result.continueThread,false);
 assert.match(result.reply,/CI/u);
 assert.match(result.reply,/тест/u);
});

test("CI failure handoff never overrides an explicit ordinary-chat test",async()=>{
 const task="Проверь строку лицензии";
 const message="В обычном чате Jessica отправь: «"+task+"». Не выдумывай результат.";
 const result=await interpretRelayTurn({message},async()=>{throw Error("unavailable")});
 assert.equal(result.action,"test");
 assert.equal(result.task,task);
});

test("blocked GitHub quote fix handoff remains a no-test reply without interpretation provider",async()=>{
 const message=[
  "JESSICA_REPLY_cb8633bf-ebb4-439c-83dc-f4e28815123f",
  "Тест не пройден: Jessica подтвердила английскую цитату, но затем перевела её на русский. Проверка не распознала метку «URL:». Запись исправления в GitHub заблокирована. Повторный тест пока не запускай.",
  "JESSICA_END_cb8633bf-ebb4-439c-83dc-f4e28815123f"
 ].join("\n");
 let calls=0;
 const reply=await interpretRelayTurn({message},async()=>{calls++;throw Error("429 exhausted");});
 assert.equal(calls,0);
 assert.equal(reply.action,"reply");
 assert.equal(reply.task,"");
 assert.equal(reply.continueThread,false);
 assert.match(reply.reply,/тест пока не запускаю/iu);
});

test("repeat in ordinary Jessica chat parses a quoted single task",async()=>{
 const task="Сравни восстановление прав по GPLv3 §8 и прекращение патентной лицензии по Apache 2.0 §3. Укажи официальные URL";
 const message="PR #92: исправление внесено, CI пройден, Render LIVE. Повтори в обычном чате Jessica: «"+task+"». Передай ответ дословно.";
 let calls=0;
 const decision=await interpretRelayTurn({message},async()=>{calls++;return "";});
 assert.equal(calls,0);
 assert.deepEqual(decision,{action:"test",reply:"",task,continueThread:false});
 assert.equal(compactJessicaTest("Не повторяй. "+message),null);
 assert.equal(compactJessicaTest(message+" Повтори в обычном чате Jessica: «Другой тест»"),null);
});

test("exact PR102 repeated zero-day test survives AI interpreter HTTP 429",async()=>{
 const id="12949dff-dee7-4489-b094-35ee723dbd19";
 const task="Сколько дней между 29.02.2024 и 29.02.2024? Исключи начальную дату, включи конечную. Покажи расчёт";
 const message=[
   "JESSICA_REPLY_"+id,
   "Исправление PR #102 опубликовано, Render LIVE. В обычном чате Jessica повтори: «"+task+"». Передай ответ дословно.",
   "JESSICA_END_"+id
 ].join("\n");
 let calls=0;
 const result=await interpretRelayTurn({message},async()=>{
   calls++;throw Error("429 quota exceeded");
 });
 assert.equal(calls,0);
 assert.deepEqual(result,{action:"test",reply:"",task,continueThread:false});
 assert.equal(compactJessicaTest("Не повторяй. "+message),null);
 assert.equal(compactJessicaTest(message+" В обычном чате Jessica повтори: «Другой тест»"),null);
});

test("zero-day explanation next step never consumes Codex interpreter quota",async()=>{
 const id="fd885daf-3ee4-4e45-9d0f-2fd69b013166";
 const body="Результат 0 верный, но расчёт не показан. Следующий шаг: исправить объяснение нулевого интервала в dateDiffResultText.js. При совпадении дат интервал пуст, учитываемых дней нет. Повторный тест пока не запускай.";
 const message=["JESSICA_REPLY_"+id,body,"JESSICA_END_"+id].join("\n");
 let calls=0;
 const result=await interpretRelayTurn({message},async()=>{
   calls++;throw Error("429 daily quota");
 });
 assert.equal(calls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.equal(result.continueThread,false);
 assert.match(result.reply,/объяснение расчёта/u);
 assert.match(result.reply,/не запускаю/u);
 const ambiguous=message.replace("Повторный тест пока не запускай.",
   "В обычном чате Jessica отправь: «Сколько дней между 01.01.2024 и 02.01.2024?»");
 const parser=(await import("../core/codex/relayPendingDateExplanation.js")).pendingZeroDayExplanationHandoff;
 assert.equal(parser(ambiguous),null);
});

test("no-test missing absolute calculation directive bypasses Codex quota",async()=>{
 const id="df5ca57f-ee5b-45a2-950b-4c1c8d37113d";
 const body="Число 3 верное, но расчёт отсутствует. Следующий шаг: исправить dateDiffResultText.js, добавить проверку полноты в Validator и регрессионный тест. Повторный тест пока не запускай.";
 const message=["JESSICA_REPLY_"+id,body,"JESSICA_END_"+id].join("\n");
 let calls=0;
 const result=await interpretRelayTurn({message,history:[]},async()=>{
   calls++;throw Error("429 exhausted provider quota");
 });
 assert.equal(calls,0);
 assert.equal(result.action,"reply");
 assert.equal(result.task,"");
 assert.equal(result.continueThread,false);
 assert.match(result.reply,/расч[её]т/u);
 assert.match(result.reply,/не запускаю/u);
 const {pendingAbsoluteDateFix}=await import("../core/codex/relayPendingAbsoluteDateFix.js");
 assert.equal(pendingAbsoluteDateFix(message.replace("Повторный тест пока не запускай.",
    "В обычном чате Jessica отправь: «Сколько будет 3 плюс 4?»")),null);
 assert.equal(pendingAbsoluteDateFix("Выполни тест в обычном чате"),null);
});

test("no-test calendar Composer fix stays deterministic on HTTP 429",async()=>{
 const id="039ed77d-246c-4920-a486-36dd18f7d993";
 const body="Ответ верный, но Composer всё ещё вызывает AI и получает HTTP 429. Следующий шаг: добавить invalid_calendar_date в directAnswer.js, исключить лишние AI-вызовы и проверить CI. Запись в GitHub заблокирована. Тест пока не запускать.";
 const message=["JESSICA_REPLY_"+id,body,"JESSICA_END_"+id].join("\n");
 let calls=0;
 const decision=await interpretRelayTurn({message,history:[]},async()=>{
  calls++;throw Error("429 exhausted provider");
 });
 assert.equal(calls,0);
 assert.equal(decision.action,"reply");
 assert.equal(decision.task,"");
 assert.equal(decision.continueThread,false);
 assert.match(decision.reply,/Тест пока не запускаю/u);
 const {pendingCalendarDirectFix}=await import("../core/codex/relayPendingCalendarDirectFix.js");
 assert.equal(pendingCalendarDirectFix(message.replace("Тест пока не запускать.",
   "В обычном чате Jessica отправь: «Новый тест».")),null);
 assert.equal(pendingCalendarDirectFix("В обычном чате Jessica отправь: «Новый тест»."),null);
});
