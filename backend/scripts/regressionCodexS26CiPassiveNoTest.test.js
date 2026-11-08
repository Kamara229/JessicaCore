import test from "node:test";
import assert from "node:assert/strict";
import {externalResearchCiHandoff} from "../core/codex/relayResearchExternalCiStatus.js";
import {interpretRelayTurn,validateRelayDecision} from "../core/codex/relayInterpreter.js";

const id="6f8ad2f6-e110-4e85-814c-7d8d2d746466";
const status="Продолжаем CALIBRI. S26 ещё обучается в GitHub. Следующий шаг — после завершения CI проверить 96 двухсловных ответов, bpp и скорость CPU. GitHub и проверки ведёт ChatGPT. Jessica пока только передаёт сообщения.";
const wrapped="JESSICA_REPLY_"+id+"\n\n"+status+"\n\nJESSICA_END_"+id;
const hallucinatedTask="после завершения CI проверить 96 двухсловных ответов, bpp и скорость CPU";

for(const message of [status,wrapped]) {
    test("real November 8 S26 CI progress never opens an ordinary Jessica test",async()=>{
        let calls=0;
        const result=await interpretRelayTurn({message,plan:"CALIBRI S26 paid 36M CI"},async()=>{
            calls++;
            return {action:"test",task:hallucinatedTask,reply:"",continueThread:false};
        });
        assert.equal(calls,0,"No quota-limited AI interpretation needed");
        assert.equal(result.action,"reply");
        assert.equal(result.task,"");
        assert.equal(result.continueThread,false);
        assert.match(result.reply,/S26/);
        assert.match(result.reply,/не запрашивался и не запускался/);
    });
    test("validator vetoes hallucinated future CI check from S26 status",()=>{
        const res=validateRelayDecision({
            action:"test",task:hallucinatedTask,reply:"",continueThread:false
        },message);
        assert.equal(res.action,"reply");
        assert.equal(res.task,"");
    });
}
test("CI status shortcut refuses explicit requests, external secrets and wrong envelopes",()=>{
    const other="00000000-0000-0000-0000-000000000000";
    for(const variant of [
        wrapped.replace("JESSICA_END_"+id,"JESSICA_END_"+other),
        status+" Запусти тест в обычном чате Jessica.",
        status+" Передай API-ключ.",
        status+" Открой https://example.com",
        status.replace("GitHub и проверки ведёт ChatGPT.",""),
        status.replace("Jessica пока только передаёт сообщения.",""),
        status.replace("после завершения CI проверить","сейчас выполнить тест"),
        hallucinatedTask,
        "Создай новый тестовый чат Jessica. Точная формулировка задачи: "+hallucinatedTask,
    ]) assert.equal(externalResearchCiHandoff(variant),null,variant);
});
test("genuine explicit Jessica test is still permitted in same interpreter",async()=>{
    const direct="Создай новый тестовый чат Jessica. Точная формулировка задачи: Сколько будет 7 умножить на 8?";
    const result=await interpretRelayTurn({message:direct},async()=>{
        throw Error("Explicit test must route without model calls");
    });
    assert.equal(result.action,"test");
    assert.equal(result.task,"Сколько будет 7 умножить на 8?");
});
