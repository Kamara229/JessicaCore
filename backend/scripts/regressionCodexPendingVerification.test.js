import test from "node:test";
import assert from "node:assert/strict";
import { interpretRelayTurn } from "../core/codex/relayInterpreter.js";
import { pendingBackendVerification } from "../core/codex/relayPendingVerification.js";

const marker = "77164669-e63a-4251-8be1-9db38e717ea4";
const latest = [
  "JESSICA_REPLY_" + marker,
  "Это не провал: страницы не переданы. В backend добавлена сборка страниц с SHA-256 и защитой от пропусков и дублей. Коммит 10c4a96. Render и CI ещё проверяются. Тест в чате пока не нужен.",
  "JESSICA_END_" + marker
].join("\n");

test("actual pending CI+Render handoff bypasses unavailable interpretation service", async () => {
  let calls = 0;
  const result = await interpretRelayTurn({ message: latest }, async () => {
    calls++; throw Error("429 quota exceeded");
  });
  assert.equal(calls, 0);
  assert.equal(result.action, "reply");
  assert.equal(result.task, "");
  assert.equal(result.continueThread, false);
  assert.match(result.reply, /Проверь CI и Render/);
  assert.match(result.reply, /Тест в обычном чате Jessica пока не нужен/);
});

test("the same progress message cannot accidentally start a Jessica test", () => {
  assert.ok(pendingBackendVerification(latest));
  assert.equal(pendingBackendVerification("Коммит 10c4a96. Render Live, CI успешно завершён. Тест пока не нужен."), null);
  assert.equal(pendingBackendVerification("Проверь работу. Render и CI ещё проверяются. Тест пока не нужен."), null);
  assert.equal(pendingBackendVerification("Коммит 10c4a96. Render и CI ещё проверяются. Запусти тест. Тест пока не нужен."), null);
  assert.equal(pendingBackendVerification("Коммит 10c4a96. Render и CI ещё проверяются. Установи APK. Тест пока не нужен."), null);
});

test("status-only handoff does not swallow an explicit test", async () => {
  const task = "Вычисли разницу между 29 октября и 6 ноября 2026 года.";
  const message = latest + "\nВ обычном чате отправь: «" + task + "»";
  const result = await interpretRelayTurn({message},async()=>{throw Error("no quota")});
  assert.equal(result.action,"test");
  assert.equal(result.task,task);
});
