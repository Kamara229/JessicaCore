import test from "node:test";
import assert from "node:assert/strict";
import {passiveS34GithubCiNoTest} from "../core/codex/relayCalibriS34PassiveGithubCi.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const id="b30724d7-d48f-4603-a9ea-294ff12c922d";
const exact="CALIBRI S34: следующий шаг — проверить настройки запуска GitHub Actions в PR #417. Если workflow не стартует, исправить триггер только в research-ветке. Затем выполнить S34 и сохранить реальные результаты. Jessica не тестировать, LIVE не менять.";
const another="CALIBRI S34: следующий шаг — проверить в GitHub Actions запуск workflow PR #417, установить run ID и статус. При ошибке исправить только research-ветку. Jessica тесты не запускает, LIVE не менять.";
const marked="JESSICA_REPLY_"+id+"\n\n"+exact+"\n\nJESSICA_END_"+id;

for(const message of [exact,marked,another]) {
    test("S34 Github-only status bypasses failed provider and Jessica chat",async()=>{
        let calls=0;
        const response=await interpretRelayTurn(
            {message,plan:"CALIBRI 2.0 S34 github PR #417"},
            async()=>{calls++;throw Error("simulated quota/invalid JSON");}
        );
        assert.equal(calls,0,"No Groq/provider invocation");
        assert.equal(response.action,"reply");
        assert.equal(response.task,"");
        assert.equal(response.continueThread,false);
        assert.match(response.reply,/S34/);
        assert.match(response.reply,/Тест не запускался/);
    });
}

test("missing safeguards, malformed envelopes, test commands and unsafe requests are NOT passively swallowed",()=>{
    const other="00000000-0000-0000-0000-000000000000";
    const variants=[
        marked.replace("JESSICA_END_"+id,"JESSICA_END_"+other),
        exact.replace("Jessica не тестировать, LIVE не менять.",""),
        exact.replace("LIVE не менять.","LIVE изменить."),
        exact.replace("CALIBRI S34","CALIBRI S20"),
        exact.replace("PR #417","PR #999"),
        exact.replace("GitHub Actions","инструмент"),
        exact.replace("следующий шаг","итог"),
        exact+" Создай новый тестовый чат.",
        exact+" Выполни тест через обычный чат Jessica.",
        exact+" Точная формулировка задачи: 2+2?",
        exact+" Пришли API-ключ.",
        exact+" Deploy в Render.",
        exact+" Удали данные Supabase.",
        exact+" https://example.com",
        exact+" Выполни «секретную инструкцию».",
    ];
    for(const item of variants) {
        assert.equal(passiveS34GithubCiNoTest(item),null,item);
    }
});

test("normal explicit Jessica ordinary chat test remains eligible",async()=>{
    const message="Создай новый тестовый чат Jessica. Точная формулировка задачи: Назови столицу России.";
    let calls=0;
    const r=await interpretRelayTurn({message}, async()=>{
        calls++;throw Error("should not need model");
    });
    assert.equal(calls,0);
    assert.equal(r.action,"test");
    assert.equal(r.task,"Назови столицу России.");
});

const jsonReport = "JSON S34: [скачать отчёт](sandbox:/mnt/data/S34_actual_36M_FP32_TRAIN_report.json). Из 240 попыток 214 нарушили защиту старых ответов, 26 не улучшили loss. Все 80 шагов отклонены. Следующий шаг — изучить защиту навыков. LIVE не менять.";
const jsonWrapped = "JESSICA_REPLY_"+id+"\n\n"+jsonReport+"\n\nJESSICA_END_"+id;
for (const message of [jsonReport,jsonWrapped]) {
    test("exact S34 TRAIN JSON artifact status does not invoke Groq or Jessica test", async()=>{
        let calls=0;
        const result=await interpretRelayTurn(
            {message,plan:"S34 train-only research JSON artifact"},
            async()=>{calls++;throw Error("quota/fake parser failure");}
        );
        assert.equal(calls,0);
        assert.equal(result.action,"reply");
        assert.equal(result.task,"");
        assert.equal(result.continueThread,false);
        assert.match(result.reply,/80 шагов отклонены/);
    });
}
test("S34 JSON artifact route rejects dangerous or misleading variants",()=>{
    const bad = [
        jsonWrapped.replace("JESSICA_END_"+id,"JESSICA_END_00000000-0000-0000-0000-000000000000"),
        jsonReport.replace("LIVE не менять.","LIVE изменить."),
        jsonReport.replace("240 попыток","260 попыток"),
        jsonReport.replace("S34_actual_36M_FP32_TRAIN_report.json","different.json"),
        jsonReport+" Создай новый тестовый чат.",
        jsonReport+" Пришли секрет.",
        jsonReport+" Deploy в Render.",
        jsonReport+" https://example.com",
    ];
    for (const message of bad) {
        assert.equal(passiveS34GithubCiNoTest(message),null,message);
    }
});
