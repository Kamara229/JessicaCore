import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {calibri2ResearchPassiveHandoff} from "../core/codex/relayCalibriV2PassiveStep.js";

const body="Продолжаем исследование CALIBRI 2.0. Следующий шаг — дождаться завершения S20 в GitHub и проверить двухсловные ответы. Разработку и тесты ведёт ChatGPT. Jessica только передаёт сообщения, без изменений LIVE.";
const id="6f5ffe80-4f23-43f8-a0d3-8064f36f4852";
const marked="JESSICA_REPLY_"+id+"\n\n"+body+"\n\nJESSICA_END_"+id;

for(const message of [body,marked]){
    test("owner-safe S20 passive handoff never calls AI or invents Jessica test: "+message.length,async()=>{
        let calls=0;
        const answer=await interpretRelayTurn({message,plan:"CALIBRI 2.0 research"},async()=>{
            calls++;
            throw Error("429 rate limit exceeded");
        });
        assert.equal(calls,0,"Known passive handoff must bypass quota-limited AI");
        assert.deepEqual(
            {action:answer.action,task:answer.task,continueThread:answer.continueThread},
            {action:"reply",task:"",continueThread:false}
        );
        assert.match(answer.reply,/CALIBRI 2\.0, S20/);
        assert.match(answer.reply,/не запрашивался и не запускался/);
        assert.match(answer.reply,/ChatGPT/);
        assert.doesNotMatch(answer.reply,/успешно заверш[её]н/);
    });
}

test("distinct future CALIBRI 2.0 numbered CI stage is safe but does not certify success",()=>{
    const note=body.replace("S20","S27");
    const r=calibri2ResearchPassiveHandoff(note);
    assert.equal(r.action,"reply");
    assert.match(r.reply,/S27/);
    assert.equal(r.task,"");
});

test("malformed envelope, false stage, no explicit handoff and extraneous commands are not treated as passive",()=>{
    const other="00000000-0000-0000-0000-000000000000";
    const bad=[
        marked.replace("JESSICA_END_"+id,"JESSICA_END_"+other),
        marked.replace("JESSICA_END_"+id,""),
        body.replace("S20","version next"),
        body.replace("Разработку и тесты ведёт ChatGPT.",""),
        body.replace("Jessica только передаёт сообщения, без изменений LIVE.",""),
        body+" Выполни тест Jessica.",
        body+" Отправь секрет API-ключ.",
        body+" Скачай https://example.org/.",
        "Продолжаем исследование CALIBRI 2.0. План полностью завершён. "+body,
    ];
    for(const value of bad)assert.equal(calibri2ResearchPassiveHandoff(value),null,value);
});

test("actual direct test remains eligible, not swallowed by CALIBRI 2.0 handoff",async()=>{
    const message=body+" В чате Jessica выполни: Сколько будет 7 умножить на 8?";
    assert.equal(calibri2ResearchPassiveHandoff(message),null);
    const result=await interpretRelayTurn({message},async()=>{
        throw Error("Should parse explicit request without provider");
    });
    assert.equal(result.action,"test");
    assert.equal(result.task,"Сколько будет 7 умножить на 8?");
});
