import assert from "node:assert/strict";
import { calibriNoTestNextStep } from "../core/codex/relayCalibriNoTestNextStep.js";
import { interpretRelayTurn } from "../core/codex/relayInterpreter.js";

const id = "0fce8736-be58-44f7-910f-ae06ec2226cf";
const body = "CALIBRI 2.0: продолжаем обучение малой модели 22,37M. Следующий шаг — проверить завершение CI #42366424415 и правильность ответов после 64 шагов. Джессика тесты не запускает. LIVE не менять.";
const envelope = `JESSICA_REPLY_${id}\n\n${body}\n\nJESSICA_END_${id}`;
let generateCalled = false;
const noProvider = async () => { generateCalled = true; throw Error("provider-called"); };

for (const message of [body, envelope]) {
  const ack = await interpretRelayTurn({
    message, plan:"CALIBRI 2.0 small-first",history:[]
  }, noProvider);
  assert.equal(ack.action,"reply");
  assert.equal(ack.task,"");
  assert.equal(ack.continueThread,false);
  assert.match(ack.reply,/CALIBRI 2\.0/u);
  assert.match(ack.reply,/Jessica тесты не запускаю/u);
  assert.equal(generateCalled,false,"No AI provider called for authorized no-test handoff");
}
// Exact no-test relay message from the second reported incident: v2-only parser
// must classify both the marker-wrapped and Android-extracted form without Groq.
const repeatId = "4a8eddfd-e013-448e-8c4d-7fb06dcd3427";
const repeatBody = "CALIBRI 2.0: следующий шаг — исследовать схлопывание ответов малой модели после 64 шагов обучения. GitHub проверяю и исправляю я. Джессика тесты не запускает, повторные подтверждения не отправляет. LIVE не менять.";
for (const message of [repeatBody,
  ["JESSICA_REPLY_"+repeatId, "", repeatBody, "", "JESSICA_END_"+repeatId].join("\n")]) {
  const ack = await interpretRelayTurn({message,plan:"CALIBRI 2.0 S0",history:[]},noProvider);
  assert.equal(ack.action,"reply");
  assert.equal(ack.task,"");
  assert.equal(ack.continueThread,false);
  assert.ok(ack.reply.includes("CALIBRI 2.0"));
  assert.equal(generateCalled,false,"Should not require AI provider for second incident");
}
const legacy = calibriNoTestNextStep(
  "CALIBRI v232: следующий шаг — проверить GitHub CI. Джессика тесты не запускает.");
assert.match(legacy,/CALIBRI v232/u);

for(const unsafe of [
  body.replace("Джессика тесты не запускает.", "Джессика запускает тест."),
  body.replace("Следующий шаг — проверить завершение CI", "Следующий шаг — запусти тест в обычном чате"),
  envelope.replace("JESSICA_END_"+id,"JESSICA_END_99999999-9999-9999-9999-999999999999")
]) assert.equal(calibriNoTestNextStep(unsafe),null,"Reject ambiguous, active-test or broken envelope");

console.log("PASS CALIBRI2 body-only and marker envelope deterministically route to safe no-test reply");
console.log("PASS original CALIBRI v232 remains supported; unsafe input not accepted");
