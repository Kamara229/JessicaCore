import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn,validateRelayDecision} from "../core/codex/relayInterpreter.js";
import {confirmedPlanCompletion} from "../core/codex/relayTerminalPolicy.js";

test("model cannot stop Codex during a healthy ongoing chat or after a test",async()=>{
    const messages=[
        "Продолжи в том же чате Джессики. Проверь следующее.",
        "Проверка прошла, продолжай план. Следующий тест: ответь КОМПАС.",
        "Системное уведомление о максимальной длине обсуждения; продолжи в выбранном чате."
    ];
    for(const message of messages) {
        for(const attemptedAction of ["finish","pause"]) {
            const got=await interpretRelayTurn({message},async()=>({
                action:attemptedAction,reply:"Всё, отключаюсь.",task:""
            }));
            assert.equal(got.action,"reply");
            assert.match(got.reply,/остаётся в выбранном чате/);
            assert.equal(got.task,"");
        }
    }
});

test("explicit completed-plan report can close a truly finished session",()=>{
    const message="План полностью завершён. Все проверки пройдены.";
    assert.equal(confirmedPlanCompletion(message),true);
    assert.equal(validateRelayDecision({action:"finish",reply:"План выполнен.",task:""},message).action,"finish");
    assert.equal(confirmedPlanCompletion("Когда план завершён, сообщи об этом."),false);
    assert.equal(confirmedPlanCompletion("Продолжи проверку. План завершён?"),false);
});

test("pause cannot end the session while requiring a safe follow-up",()=>{
    const got=validateRelayDecision({action:"pause",reply:"Нужен новый чат, отключаюсь.",task:""},"Выполни следующий текстовый тест.");
    assert.equal(got.action,"reply");
});
