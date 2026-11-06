import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {calibriNoTestNextStep} from "../core/codex/relayCalibriNoTestNextStep.js";

// The exact last relay that previously fell through to a quota-bound LLM
// and returned a false-negative "service of interpretation failed" incident.
const body="CALIBRI v211: CI выполняет сравнение 48 полных TRAIN-ответов. Следующий шаг — определить потери правильных ответов и повторы между v206 и v208. Результаты ещё не получены. Jessica не тестировать; LIVE не менять.";
const uuid="508fa39f-4083-49ad-b46b-040f88861b0c";
const envelope="JESSICA_REPLY_"+uuid+"\n\n"+body+"\n\nJESSICA_END_"+uuid;

for(const message of [body,envelope]){
    test("no-test frozen CALIBRI research relay bypasses Groq: "+message.length, async()=>{
        let quotaCalls=0;
        const decision=await interpretRelayTurn({message,plan:"CALIBRI v211"},
            async()=>{quotaCalls++;throw Error("429 quota");});
        assert.equal(quotaCalls,0,"A no-test CALIBRI progress update never needs an LLM");
        assert.equal(decision.action,"reply");
        assert.equal(decision.task,"");
        assert.equal(decision.continueThread,false);
        assert.match(decision.reply,/v211/);
        assert.match(decision.reply,/тесты не запускаю/);
        assert.doesNotMatch(decision.reply,/сбой|успешный тест/i);
    });
}

test("strict boundary: no invented test from vague CALIBRI status",()=>{
    assert.equal(calibriNoTestNextStep(body.replace("Jessica не тестировать","")),
        null);
    assert.equal(calibriNoTestNextStep(envelope.replace("JESSICA_END_"+uuid,
        "JESSICA_END_00000000-0000-0000-0000-000000000000")),null);
    assert.equal(calibriNoTestNextStep(body+" Используй секрет GitHub."),null);
    assert.equal(calibriNoTestNextStep("Выполни тест Jessica. "+body),null);
});

test("an explicit chat test must not be reclassified as passive", async()=>{
    const message="CALIBRI v211: сравнение TRAIN. Jessica не тестировать. "+
      "Создай новый тестовый чат Jessica. Точная формулировка: Сколько будет 2+2?";
    let interpreted=0;
    const decision=await interpretRelayTurn({message},async()=>{
        interpreted++;
        return {action:"test",reply:"",task:"Сколько будет 2+2?",continueThread:false};
    });
    assert.equal(decision.action,"test");
    assert.equal(decision.task,"Сколько будет 2+2?");
    assert.ok(interpreted<=1);
});
