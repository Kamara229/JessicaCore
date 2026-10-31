import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {numberedBenchmarkTest} from "../core/codex/relayNumberedBenchmarkTest.js";

const task = "Есть 8 одинаковых на вид шаров. Один тяжелее остальных. Есть чашечные весы без гирь. Как найти тяжёлый шар за два взвешивания? Объясни все варианты.";
const message = [
  "JESSICA_REPLY_a149d1c0-3961-4217-afe4-ed66766f9827",
  "Тест 4/60 пройден. Ответ правильный.",
  "",
  "Тест 5/60: "+task,
  "",
  "JESSICA_END_a149d1c0-3961-4217-afe4-ed66766f9827"
].join("\n");

test("numbered reasoning benchmark bypasses exhausted Codex interpretation quota", async () => {
  let providerCalls = 0;
  const result = await interpretRelayTurn({message,history:[]}, async () => {
    providerCalls++;
    throw Error("429 provider daily quota exhausted");
  });
  assert.equal(providerCalls,0);
  assert.deepEqual(result,{action:"test",reply:"",task,continueThread:false});
});

test("numbered benchmark parser is exact and rejects examples, negation and multiple tasks", () => {
  assert.equal(numberedBenchmarkTest(message)?.task,task);
  assert.equal(numberedBenchmarkTest("Тест 4/60 пройден. Ответ правильный."),null);
  assert.equal(numberedBenchmarkTest("Например:\nТест 5/60: "+task),null);
  assert.equal(numberedBenchmarkTest("Пример:\nТест 5/60: "+task),null);
  assert.equal(numberedBenchmarkTest("Тест 5/60: "+task+"\nНе запускай тест."),null);
  assert.equal(numberedBenchmarkTest("Тест 5/60: "+task+"\nТест 6/60: "+task),null);
  assert.equal(numberedBenchmarkTest("Тест 61/60: "+task),null);
  assert.equal(numberedBenchmarkTest("Тест 5/60: 2+2"),null);
});

// Keep variant-specific cases small while running them in the existing relay CI.
import "./regressionCodexInlineBenchmark.test.js";
