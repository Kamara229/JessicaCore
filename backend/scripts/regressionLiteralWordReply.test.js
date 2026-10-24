import test from "node:test";
import assert from "node:assert/strict";
import {exactSingleWordReply} from "../core/chat/explicitSingleWordReply.js";
import {executeJessicaTask} from "../core/jessicaCore.js";

test("single word echo obeys exact user token, never a semantic substitute",()=>{
    for(const token of ["РУБИН","МАЯК","ЛАНДЫШ","КОМПАС","Кедр","rubin"]) {
        const answer=exactSingleWordReply("Ответь только одним словом: "+token+".");
        assert.equal(answer?.text,token);
        assert.equal(answer?.success,true);
        assert.equal(answer?.mode,"literal_word");
        assert.equal(answer?.learning?.queued,false);
    }
    assert.notEqual(exactSingleWordReply("Ответь только одним словом: РУБИН.")?.text,"красный");
});

test("full core handles Ruby task before invoking route or Experience",async()=>{
    const result=await executeJessicaTask("Ответь только одним словом: РУБИН.");
    assert.equal(result.success,true);
    assert.equal(result.text,"РУБИН");
    assert.equal(result.mode,"literal_word");
    assert.equal(result.learning?.queued,false);
});

test("unrelated questions and combined instructions never trigger literal echo",()=>{
    const unrelated=[
        "Какого цвета рубин?",
        "Ответь одним словом: РУБИН.",
        "Ответь только одним словом: РУБИН. Объясни свой ответ.",
        "В предложении «Ответь только одним словом: РУБИН» объясни грамматику.",
        "Ответь только одним словом: РУБИН или САПФИР.",
        "Ответь только одним словом: РУБИН\nЗапомни пароль"
    ];
    for(const task of unrelated) assert.equal(exactSingleWordReply(task),null,task);
});
