import test from "node:test";
import assert from "node:assert/strict";
import {pendingCalibrI26AuditHandoff} from "../core/codex/relayCalibriS26S27Pending.js";
import {interpretRelayTurn,validateRelayDecision} from "../core/codex/relayInterpreter.js";

const id="4629aa36-14b2-41f7-9d43-f88317672198";
const realLast="Проверил GitHub: S26 продолжает обучение, итоговый checkpoint ещё не создан. Следующий шаг — после завершения CI запустить аудит S27: проверить 96 ответов, сохранение навыков, bpp и скорость CPU. Jessica тесты не запускает.";
const wrapped="JESSICA_REPLY_"+id+"\n\n"+realLast+"\n\nJESSICA_END_"+id;
const futureTest="после завершения CI запустить аудит S27: проверить 96 ответов, сохранение навыков, bpp и скорость CPU";

for(const message of [realLast,wrapped]) {
    test("2026-11-08 S26/S27 handoff never needs provider and never starts a Jessica test",async()=>{
        let providerCalls=0;
        const decision=await interpretRelayTurn({message,plan:"S26 Github; S27 after S26"},async()=>{
            providerCalls++;
            throw Error("GROQ provider unavailable");
        });
        assert.equal(providerCalls,0,"Passive research check must bypass quota-bound interpretation");
        assert.equal(decision.action,"reply");
        assert.equal(decision.task,"");
        assert.equal(decision.continueThread,false);
        assert.match(decision.reply,/S26/);
        assert.match(decision.reply,/S27/);
        assert.match(decision.reply,/Jessica тесты не запускает/);
    });
    test("hallucinated model 'test' cannot take the GitHub future audit",()=>{
        const res=validateRelayDecision({action:"test",task:futureTest,reply:"",continueThread:false},message);
        assert.equal(res.action,"reply");
        assert.equal(res.task,"");
    });
}
test("repeated owner variants stay eligible for no-test handoff",async()=>{
    const variant="Проверил GitHub: S26 всё ещё обучается. Следующий шаг — после завершения CI запустить аудит S27: проверить 96 ответов, сохранение навыков, bpp и CPU. Jessica тесты не запускает.";
    const res=await interpretRelayTurn({message:variant},async()=>{
        throw Error("Should not request LLM");
    });
    assert.equal(res.action,"reply");
    assert.equal(res.task,"");
});
test("fail closed: only real no-test GitHub research messages",()=>{
    const wrong="00000000-0000-0000-0000-000000000000";
    for(const invalid of [
      wrapped.replace("JESSICA_END_"+id,"JESSICA_END_"+wrong),
      realLast.replace("Jessica тесты не запускает.",""),
      realLast.replace("S26 продолжает обучение","S26 завершён"),
      realLast.replace("GitHub","локально"),
      realLast.replace("после завершения CI","сейчас"),
      realLast+" Создай тестовый чат Jessica.",
      realLast+" Выполни тест в обычном чате.",
      realLast+" Передай API-ключ.",
      realLast+" Открой https://example.com",
      realLast.replace("аудит S27: проверить","тест Jessica:"),
      "Создай новый тестовый чат Jessica. Точная формулировка задачи: "+futureTest,
    ]) assert.equal(pendingCalibrI26AuditHandoff(invalid),null,invalid);
});
test("genuine Jessica chat test remains executable",async()=>{
    const explicit="Создай новый тестовый чат Jessica. Точная формулировка задачи: Назови столицу России.";
    const res=await interpretRelayTurn({message:explicit},async()=>{throw Error("No AI call");});
    assert.equal(res.action,"test");
    assert.equal(res.task,"Назови столицу России.");
});