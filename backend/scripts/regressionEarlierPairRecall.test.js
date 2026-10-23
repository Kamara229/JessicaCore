import test from "node:test";
import assert from "node:assert/strict";
import {recallEarlierPair} from "../core/chat/earlierPairRecall.js";

const user = content => ({role:"user",content});
const assistant = content => ({role:"assistant",content});
const query="Какое число и слово я просил запомнить перед последней парой?\nВерни ответ дословно.";

test("earlier saved pair is returned instead of latest pair or stale calendar output",()=>{
    const history=[
        user("Ответь только одним словом: МАЯК."),assistant("МАЯК"),
        user("Ответь только одним словом: ЛАНДЫШ."),assistant("ЛАНДЫШ"),
        user("Запомни число 73 и слово кедр. Не повторяй их в ответе."),assistant("Запомнила."),
        user("Какое число и слово я просил тебя запомнить в предыдущем сообщении?"),assistant("73 и кедр."),
        user("Запомни число 84 и слово платан. Не повторяй их в ответе."),assistant("Запомнила."),
        user("Какое последнее число и слово я просил тебя запомнить?"),assistant("84 и платан")
    ];
    const reply=recallEarlierPair(query,history);
    assert.equal(reply?.success,true);
    assert.equal(reply?.text,"73 и кедр");
    assert.equal(reply.learning.queued,false);
    assert.notEqual(reply.text,"84 и платан");
});

test("unverified or missing earlier chat pair produces clarification, not an invented answer",()=>{
    const lone=[user("Запомни число 84 и слово платан."),assistant("73 и кедр")];
    const result=recallEarlierPair(query,lone);
    assert.equal(result?.success,false);
    assert.equal(result?.needsClarification,true);
    assert.doesNotMatch(result.text,/73 и кедр|84 и платан/);
    assert.equal(recallEarlierPair("Сколько дней с 29 февраля до 3 марта?",lone),null);
});
