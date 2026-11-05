import test from "node:test";
import assert from "node:assert/strict";
import {safeCalibriPassiveHandoff} from "../core/codex/relayCalibriPassiveSafe.js";
import {registerCodexRoutes} from "../http/routes/codexRoute.js";

const cases=[
    "Продолжаем CALIBRI v181. Следующий шаг: передавай только новые запросы пользователя. Подтверждения не требуются. Исследование, код и тесты в GitHub выполняет ChatGPT самостоятельно.",
    "CALIBRI v181 выполняется в GitHub. Следующий шаг: прекрати отправлять повторные подтверждения. Оставайся посредником и передавай только новые сообщения пользователя дословно. Тесты и разработку ведёт ChatGPT.",
    "CALIBRI v181: испытание ещё выполняется. Следующий шаг — дождаться результатов GitHub, которые ChatGPT проверит самостоятельно. Кодекс остаётся в текущем чате. Ничего не запускай и не изменяй.",
    "CALIBRI v180: Исследование и тестирование ведёт ChatGPT. Джессика тесты не запускает.",
];
const uuid="8f8d29d8-edbb-4a5c-a08f-4202f49f91e2";
for(const body of cases)for(const message of [
    body,
    "JESSICA_REPLY_"+uuid+"\n"+body+"\nJESSICA_END_"+uuid,
]){
    test("passive CALIBRI v181 no-test message: "+message.length,()=>{
        const result=safeCalibriPassiveHandoff(message);
        assert.ok(result);
        assert.equal(result.action,"reply");
        assert.equal(result.task,"");
        assert.equal(result.continueThread,false);
        assert.match(result.reply,/не запрошен и не запускался/);
    });
}
test("HTTP relay route handles passive status without AI provider",async()=>{
    const routes=new Map();
    registerCodexRoutes({post:(path,...handlers)=>routes.set(path,handlers.at(-1))});
    let result;
    const response={set:()=>response,json:(x)=>{result=x;return response},
        status:()=>response};
    await routes.get("/api/codex/relay/interpret")({body:{message:cases[0]}},response);
    assert.equal(result.success,true);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
});
test("explicit test or malformed envelope is not swallowed",()=>{
    const blocked=[
        "CALIBRI v181: Ничего не запускай. Затем проведи тест в обычном чате Jessica.",
        "CALIBRI v181: Запусти тест в обычном чате Jessica.",
        "CALIBRI v181: Исследование продолжается.",
        "Другой проект: "+cases[0],
        "JESSICA_REPLY_"+uuid+"\n"+cases[0],
        "JESSICA_REPLY_"+uuid+"\n"+cases[0]+"\nJESSICA_END_00000000-0000-0000-0000-000000000000",
    ];
    for(const input of blocked)assert.equal(safeCalibriPassiveHandoff(input),null);
});
