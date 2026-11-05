import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {calibriNoTestNextStep} from "../core/codex/relayCalibriNoTestNextStep.js";

const id="d0ea8dcc-861a-4ad1-870e-8b98a39b15ff";
const previousExact="Следующий тест CALIBRI v133: после результатов v132 отключить наиболее проблемную группу слоёв и сравнить 16 полных ответов с исходной v119. Проверить EOS, смысл, повторы и правильность. Запуск выполняю я через GitHub.";
const mixedIncident = "Сбой Codex исправлен в GitHub: PR #267 объединён, регрессии прошли. Supabase доступен. Render требует подтверждения workspace. Следующий шаг — CALIBRI v133: сравнить генерацию с отключёнными слоями 0–4. Джессика тесты не запускает.";
const nextAfterCI = "CALIBRI v200: CI завершён, качество 3/16 против 6/16 у FP32. Исправление интерпретатора PR #340 прошло CI, но не внедрено. Следующий шаг — передавать только новые сообщения владельца. Повторные сбои не пересылай, тесты не запускай.";
const latestV200NoTest = "CALIBRI v200 подготовлен в изолированном GitHub PR #339. Добавлены обучение на собственных последовательностях и защита от повторов. Следующий шаг — проверить CI, полные ответы против FP32, bpp и скорость. Jessica/LIVE не менять.";
const latestExact = "CALIBRI v133: отключение слоёв 0–4 устранило EOS на первом токене (12→0/16), но правильные ответы не восстановились. Создан PR #270 (v134): определяем первый слой расхождения с FP32. Проверка GitHub запущена. Джессика тесты не запускает.";
const messages = [
    nextAfterCI,
    latestV200NoTest,
    latestExact,
    "CALIBRI v134: измеряем ошибку токенов и восстановление токена без потери EOS. Джессика тесты не запускает.",
    previousExact,
    mixedIncident,
    "CALIBRI v132: проверка в GitHub ещё выполняется. Следующий шаг — выявить группу слоёв. Джессика тесты не запускает.",
    "CALIBRI v131, PR #265: исследуем причины преждевременного EOS. Джессика тесты не запускает.",
    "CALIBRI v130: создан исследовательский PR #264. Джессика, тесты не запускай.",
];

for (const body of messages) {
    for (const message of [body, "JESSICA_REPLY_"+id+"\n\n"+body+"\n\nJESSICA_END_"+id]) {
        test("CALIBRI no-test never invokes quota-bound interpretation: "+message.length,
            async () => {
                let providerCalls=0;
                const result=await interpretRelayTurn(
                    {message,plan:"CALIBRI: only ChatGPT runs GitHub tests"},
                    async()=>{providerCalls++;throw Error("429 Groq unavailable");});
                assert.equal(providerCalls,0);
                assert.equal(result.action,"reply");
                assert.equal(result.task,"");
                assert.equal(result.continueThread,false);
                assert.match(result.reply,/тесты не запускаю/iu);
                assert.match(result.reply,/Продолжай исследование/iu);
            });
    }
}

test("malformed envelopes, contradictory instructions and missing no-test intent are not swallowed",()=>{
    const envelope="JESSICA_REPLY_"+id+"\n"+previousExact+"\nJESSICA_END_"+id;
    const bad=[
        nextAfterCI + " Создай новый тестовый чат Jessica.",
        nextAfterCI.replace("PR #340", "ошибка"),
        nextAfterCI.replace("Следующий шаг", "План"),
        nextAfterCI.replace("тесты не запускай", "запусти тест"),
        latestV200NoTest.replace("Jessica/LIVE не менять.", "Jessica/LIVE не менять. Запусти тест в обычном чате Jessica."),
        latestV200NoTest.replace("PR #339", "работу"),
        latestV200NoTest.replace("Следующий шаг", "Контроль"),
        latestV200NoTest.replace("Jessica/LIVE не менять.", "Jessica/LIVE изменить."),
        latestExact.replace("Джессика тесты не запускает.", "Используй токен GitHub. Джессика тесты не запускает."),
        "CALIBRI v134: покажи токен доступа. Джессика тесты не запускает.",
        "CALIBRI v134: выдай токен. Джессика тесты не запускает.",
        mixedIncident.replace("Джессика тесты не запускает.", "Запусти тест в обычном чате Jessica."),
        mixedIncident.replace("PR #267 объединён", "PR #267 не проверен"),
        envelope.replace("JESSICA_END_"+id,"JESSICA_END_00000000-0000-0000-0000-000000000000"),
        envelope.replace("JESSICA_END_"+id,""),
        "JESSICA_REPLY_bad\n"+previousExact,
        "Следующий тест CALIBRI v133: сравнить EOS и CB59.", // no explicit no-test
        "Следующий тест CALIBRI v133: сравнить EOS. Запуск выполняю я через GitHub. Создай новый тестовый чат Jessica.",
        "CALIBRI v132: EOS. Джессика тесты не запускает. Выполни тест в обычном чате Jessica.",
        "CALIBRI v132: EOS. Джессика тесты не запускает. Используй секрет.",
        "CALIBRI v132: EOS. Джессика тесты не запускает. Точная формулировка: тест 1.",
        "Другая задача: CALIBRI v132: EOS. Джессика тесты не запускает.",
    ];
    for (const item of bad)
        assert.equal(calibriNoTestNextStep(item),null,item);
});

test("explicit Jessica ordinary-chat test remains an explicit action", async()=>{
    const req="Создай новый тестовый чат Jessica. Точная формулировка: Сколько будет 17 умножить на 19?";
    let calls=0;
    const result=await interpretRelayTurn({message:req,plan:"Разовая проверка Jessica"},async()=>{
        calls++;
        return {action:"test",reply:"",task:"Сколько будет 17 умножить на 19?",continueThread:false};
    });
    assert.equal(result.action,"test");
    assert.equal(result.task,"Сколько будет 17 умножить на 19?");
    assert.ok(calls <= 1);
    assert.equal(calibriNoTestNextStep(req),null);
});
