import test from "node:test";
import assert from "node:assert/strict";
import {nonTestCiTaskHandoff} from "../core/codex/relayNonTestCiTask.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const message="Дословная задача: «анализ результатов v95».\n\n"+
    "Это проверка результатов GitHub Actions, а не текстовый тест в чате Jessica. Запуск теста Jessica не требуется.";

test("one exact non-test task never invokes AI interpreter nor Jessica test", async()=>{
    let calls=0;
    const result=await interpretRelayTurn({message},async()=>{calls++;throw Error("HTTP 429 provider down");});
    assert.equal(calls,0);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
    assert.equal(result.continueThread,false);
    assert.match(result.reply,/не тест Jessica/);
    assert.match(result.reply,/тест Jessica не запускаю/);
});
test("exact wrapper and stripped body both accepted, mismatched marker rejected",()=>{
    const uuid="f28679ac-aa9d-407e-9ab6-271ec1147eb0";
    const enveloped="JESSICA_REPLY_"+uuid+"\n\n"+message+"\n\nJESSICA_END_"+uuid;
    assert.equal(nonTestCiTaskHandoff(enveloped)?.action,"reply");
    assert.equal(nonTestCiTaskHandoff(message)?.action,"reply");
    assert.equal(nonTestCiTaskHandoff(enveloped.replace("JESSICA_END_","JESSICA_END_dead")) ,null);
});
test("explicit Jessica test requests and contradictory wording never bypass interpreter",()=>{
    for(const unsafe of [
        "Выполни тест в обычном чате Jessica. "+message,
        message.replace("не требуется","требуется"),
        message.replace("а не текстовый","и текстовый"),
        "Дословная задача: «Сколько будет 2 + 2?»."+message,
        "Дословная задача: «анализ результатов v95». Запусти тест в чате Jessica.",
        "Дословная задача: «анализ результатов v95». "+message
    ])assert.equal(nonTestCiTaskHandoff(unsafe),null);
});
test("fail closed on arbitrary command, missing full provenance or malformed marker",()=>{
    for(const bad of [
        "",
        "Дословная задача: «анализ результатов v95».",
        "Дословная задача: «обнови секретный ключ Supabase». "+message,
        "JESSICA_REPLY_x\n"+message+"\nJESSICA_END_x",
        message+"\nJESSICA_END_f28679ac-aa9d-407e-9ab6-271ec1147eb0",
    ])assert.equal(nonTestCiTaskHandoff(bad),null);
});
