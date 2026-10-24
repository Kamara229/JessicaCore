import test from "node:test";
import assert from "node:assert/strict";
import {executeJessicaTask} from "../core/jessicaCore.js";
import {recallEarlierPair} from "../core/chat/earlierPairRecall.js";

const user=content=>({role:"user",content});
const bot=content=>({role:"assistant",content});
const history=[
    user("Ответь только одним словом: РУБИН."),bot("РУБИН"),
    user("Напиши строго одно слово: САПФИР."),bot("САПФИР"),
    user("Выведи ровно одно слово: ИЗУМРУД."),bot("ИЗУМРУД"),
    user("Какое слово я просил вывести в предыдущей задаче?"),bot("ИЗУМРУД")
];
const prompt="Какое слово я просил написать строго одним словом перед командой вывести ИЗУМРУД?\nВерни ответ дословно.";

test("user's earlier Sapphire command resolves from same-chat history",async()=>{
    const recall=recallEarlierPair(prompt,history);
    assert.equal(recall?.success,true);
    assert.equal(recall?.text,"САПФИР");
    assert.equal(recall?.learning?.queued,false);
    assert.equal((await executeJessicaTask(prompt,{conversation:history})).text,"САПФИР");
});
test("empty or another chat cannot invent a remembered command",async()=>{
    for (const context of [[],[user("Выведи ровно одно слово: ИЗУМРУД."),bot("САПФИР")]]) {
        const answer=recallEarlierPair(prompt,context);
        assert.equal(answer?.success,false);
        assert.match(answer?.text || "",/нет подтверждения/);
        assert.notEqual(answer.text,"САПФИР");
    }
});
test("the same policy handles another unrelated earlier word",()=>{
    const context=[user("Напиши строго одно слово: ОПАЛ."),bot("ОПАЛ"),
        user("Выведи ровно одно слово: ГРАНАТ."),bot("ГРАНАТ")];
    const answer=recallEarlierPair("Какое слово я просил написать строго одним словом перед командой вывести ГРАНАТ?",context);
    assert.equal(answer?.text,"ОПАЛ");
});
