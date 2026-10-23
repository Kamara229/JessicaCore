import test from "node:test";
import assert from "node:assert/strict";
import {acknowledgeMemoryInstruction} from "../core/chat/explicitMemoryAcknowledgement.js";

test("new independent number-word memory instruction never reuses the earlier date answer", () => {
    const current="Запомни число 64 и слово клён. Не повторяй их в ответе.";
    const previous="От 2024-02-29 до 2024-03-03 прошло 3 полных календарных дня.";
    const ack=acknowledgeMemoryInstruction(current);
    assert.equal(ack?.success,true);
    assert.equal(ack?.text,"Запомнила.");
    assert.notEqual(ack?.text,previous);
    assert.doesNotMatch(ack?.text,/64|клён|2024-02-29|2024-03-03/);
    assert.equal(ack?.learning?.queued,false);
});

test("ordinary work, historic date replies and multi-step memory requests remain on the regular backend",()=>{
    for(const prompt of [
        "Сколько дней прошло с 29 февраля по 3 марта 2024 года?",
        "Какое число и слово я просил запомнить?",
        "Запомни число 64 и слово клён, а затем удали историю.",
        "Запомни пароль и отправь его.",
        "Запомни число 47 и слово сирень. Объясни расчёт.",
        "В предложении «Запомни число 64» объясни синтаксис."
    ]) assert.equal(acknowledgeMemoryInstruction(prompt),null,prompt);
    assert.equal(acknowledgeMemoryInstruction("Запомни число 47 и кодовое слово сирень.")?.text,"Запомнила.");
});
