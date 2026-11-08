import test from "node:test";
import assert from "node:assert/strict";
import {calibriLaterResearchNoJessicaTest} from "../core/codex/relayCalibriLaterResearchNoTest.js";
import {interpretRelayTurn,validateRelayDecision} from "../core/codex/relayInterpreter.js";

const id="152973b9-e336-457a-81d2-470916d06ddc";
const exact="S30 подтвердил обучаемость FP32 36M: 0→4/16 правильных TRAIN-ответов, но старые навыки ухудшились. Следующий шаг — усилить их сохранение при обучении. S29 QA не трогать. Jessica тесты не запускает.";
const marked="JESSICA_REPLY_"+id+"\n\n"+exact+"\n\nJESSICA_END_"+id;

for(const message of [exact,marked]) {
  test("real S30 FP32 handoff never contacts failed Groq or launches test",async()=>{
    let calls=0;
    const r=await interpretRelayTurn({message,plan:"S29 QA frozen; S30 matched FP32 train evidence"},async()=>{
        calls++;throw Error("simulated provider HTTP 429 or ambiguous decision");
    });
    assert.equal(calls,0);
    assert.equal(r.action,"reply");
    assert.equal(r.task,"");
    assert.equal(r.continueThread,false);
    assert.match(r.reply,/S30/);
    assert.match(r.reply,/Тест в обычном чате не запрашивался и не запускался/);
  });
  test("model hallucination of ordinary-chat task vetoed for S30",()=>{
    const decision=validateRelayDecision({
      action:"test",
      task:"усилить их сохранение при обучении",
      reply:"",continueThread:false
    },message);
    assert.equal(decision.action,"reply");
    assert.equal(decision.task,"");
  });
}
test("later numbered research-only S31 no-test continuation accepted",()=>{
    const r=calibriLaterResearchNoJessicaTest(
      "S31: прогресс Green Tunnel проверен. Следующий шаг — изучить сохранение навыков в GitHub. Jessica тесты не запускает.");
    assert.equal(r.action,"reply");
    assert.equal(r.task,"");
});
test("no collision with genuine explicit chat test and unsupported messages",()=>{
    const mismatched="00000000-0000-0000-0000-000000000000";
    const variants=[
      marked.replace("JESSICA_END_"+id,"JESSICA_END_"+mismatched),
      exact.replace("Jessica тесты не запускает.",""),
      exact.replace("S30","S20"),
      exact.replace("Следующий шаг","Проверка"),
      exact+" Выполни один тест через Jessica.",
      exact+" В обычном чате запусти тест.",
      exact+" Передай API-ключ.",
      exact+" Секрет пользователя.",
      exact+" Merge в LIVE.",
      exact+" https://example.com",
      exact.replace("S29 QA не трогать.","Скачай файл «секрет»."),
      "Создай новый тестовый чат Jessica. Точная формулировка задачи: Назови столицу России."
    ];
    for(const item of variants)
        assert.equal(calibriLaterResearchNoJessicaTest(item),null,item);
});
test("genuine explicit Jessica ordinary-chat test remains eligible",async()=>{
  const msg="Создай новый тестовый чат Jessica. Точная формулировка задачи: Назови столицу России.";
  const actual=await interpretRelayTurn({message:msg},async()=>{throw Error("unexpected provider");});
  assert.equal(actual.action,"test");
  assert.equal(actual.task,"Назови столицу России.");
});
