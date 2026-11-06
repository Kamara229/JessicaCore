import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {calibriNoTestNextStep} from "../core/codex/relayCalibriNoTestNextStep.js";

const id="d0ea8dcc-861a-4ad1-870e-8b98a39b15ff";
const previousExact="Следующий тест CALIBRI v133: после результатов v132 отключить наиболее проблемную группу слоёв и сравнить 16 полных ответов с исходной v119. Проверить EOS, смысл, повторы и правильность. Запуск выполняю я через GitHub.";
const mixedIncident = "Сбой Codex исправлен в GitHub: PR #267 объединён, регрессии прошли. Supabase доступен. Render требует подтверждения workspace. Следующий шаг — CALIBRI v133: сравнить генерацию с отключёнными слоями 0–4. Джессика тесты не запускает.";
const latestExact = "CALIBRI v133: отключение слоёв 0–4 устранило EOS на первом токене (12→0/16), но правильные ответы не восстановились. Создан PR #270 (v134): определяем первый слой расхождения с FP32. Проверка GitHub запущена. Джессика тесты не запускает.";
const messages = [
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


const v220Exact = "CALIBRI v220: CI #42298991456 ещё работает. Следующий шаг — проверить завершение обучения и SHA256 сжатых весов, затем сравнить правильные ответы, bpp и скорость CPU с v206. Jessica не тестировать, LIVE не менять.";

for (const message of [v220Exact, "JESSICA_REPLY_"+id+"\n\n"+v220Exact+"\n\nJESSICA_END_"+id]) {
    test("v220 exact 2026-11-06 no-test handoff avoids quota-bound provider: "+message.length,
        async () => {
            let providerCalls=0;
            const result=await interpretRelayTurn(
                {message,plan:"Only GitHub research; Jessica and LIVE unchanged"},
                async()=>{providerCalls++;throw Error("429 Groq unavailable");});
            assert.equal(providerCalls,0,"no unexpected model interpretation");
            assert.equal(result.action,"reply");
            assert.equal(result.task,"");
            assert.equal(result.continueThread,false);
            assert.match(result.reply,/тесты не запускаю/iu);
        });
}


const v220LatestExact = "CALIBRI v220: все 16 ответов уже извлечены. Следующий шаг — исследовать потерю правильного Yes на вопросе 24 и повторение цифры 5 на вопросе 0, затем исправить Green Tunnel. GitHub веду сам. Jessica не тестировать, LIVE не менять.";
for (const message of [v220LatestExact, "JESSICA_REPLY_"+id+"\\n"+v220LatestExact+"\\nJESSICA_END_"+id]) {
    test("v220 latest read-only handoff is parsed without quota-bound provider: "+message.length,
        async () => {
            let providerCalls=0;
            const result=await interpretRelayTurn(
                {message,plan:"CALIBRI source restoration is isolated, Jessica remains a passive relay"},
                async()=>{providerCalls++;throw Error("upstream unavailable");});
            assert.equal(providerCalls,0);
            assert.equal(result.action,"reply");
            assert.equal(result.task,"");
            assert.equal(result.continueThread,false);
            assert.match(result.reply,/тесты не запускаю/iu);
        });
}
