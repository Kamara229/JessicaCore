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
        exact.replace("Следующий шаг","Итог"),
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
