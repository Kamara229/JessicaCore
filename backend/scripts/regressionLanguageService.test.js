import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    normalizeOutputLanguage,
    buildLanguageInstructions,
    localizeResponse,
    protectedAnswerTokens
} from "../core/language/responseLanguageService.js";

test("only Russian and English are valid interface locales", () => {
    assert.equal(normalizeOutputLanguage("en"),"en");
    assert.equal(normalizeOutputLanguage("ru"),"ru");
    assert.equal(normalizeOutputLanguage("fr"),"ru");
    assert.equal(normalizeOutputLanguage("EN"),"ru");
    assert.equal(normalizeOutputLanguage(null),"ru");
});

test("language instructions preserve translation target, names and citations", () => {
    const policy=buildLanguageInstructions("en");
    for(const word of ["ANY language","explicitly asks for a translation","person names",
        "source citations","exact document quotations","never reject"]) {
        assert.ok(policy.includes(word), word);
    }
});

test("English UI rephrases only final displayed text, not learning and verification", async () => {
    const source={
        text:"Готово: проект Jessica Core, 2026. https://example.com/source",
        success:true,mode:"single", validated:true, learning:{queued:true},
        executionTrace:{status:"COMPLETED"}, toolResults:[{text:"original"}]
    };
    let received=null;
    const answer=await localizeResponse(source,"Summarize in English", "en",
        async (prompt,data)=>{received={prompt,data};return "Done: Jessica Core, 2026. https://example.com/source"});
    assert.equal(answer.text,"Done: Jessica Core, 2026. https://example.com/source");
    assert.equal(answer.outputLanguage,"en");
    assert.equal(answer.localizationStatus,"adapted");
    assert.equal(source.text.startsWith("Готово"),true);
    assert.equal(answer.executionTrace,source.executionTrace);
    assert.equal(answer.learning,source.learning);
    assert.equal(answer.toolResults,source.toolResults);
    assert.equal(received.data.userTask,"Summarize in English");
});

test("multilingual task destination is respected by instructions", () => {
    const p=buildLanguageInstructions("ru");
    assert.ok(p.includes("THAT requested language"));
    assert.ok(p.includes("non-UI world languages"));
});

test("document diagnostic identifier stays stable and hash matches rendered answer", async () => {
    const document={
        mode:"document", success:true, responseId:"JDOC-ABCDE12345",
        responseSha256:"old-digest",text:"Анализ. Код ответа: JDOC-ABCDE12345"
    };
    const changed=await localizeResponse(document,"Review", "en",
        async ()=>"Analysis. Response ID: JDOC-ABCDE12345");
    assert.equal(changed.responseId,document.responseId);
    assert.equal(changed.responseSha256,createHash("sha256").update(changed.text).digest("hex"));
    const omitted=await localizeResponse(document,"Review", "en",
        async ()=>"Analysis complete");
    assert.equal(omitted.text,document.text);
    assert.equal(omitted.localizationStatus,"protected-token-mismatch");
});

test("translation outage and oversize responses preserve the original factual text", async () => {
    const response={success:true,text:"Оригинальный ответ"};
    const fallback=await localizeResponse(response,"Task","en",async()=>{throw new Error("offline")});
    assert.equal(fallback.text,response.text);
    const large=await localizeResponse({...response,text:"A".repeat(14001)},"task","en",
        async ()=>{throw new Error("must not call")});
    assert.equal(large.localizationStatus,"long-answer-original");
});

test("source quotations and protected names stay intact in multilingual legal answers", async () => {
    const source={
        success:true,mode:"document",evidenceChecked:true,text:
        "АО «АЛМАЗ» сообщает: «Договор подписан 21.10.2026». Ссылка: https://example.org/letter",
        responseId:"JDOC-TEST1",responseSha256:"before"
    };
    const revised=await localizeResponse(source,"Summarize in English","en",
        async()=>"ALMAZ Inc. reports that the contract was signed on October 21, 2026. https://example.org/letter");
    assert.equal(revised.localizationStatus,"protected-token-mismatch");
    assert.equal(revised.text,source.text);
    assert.ok(protectedAnswerTokens(source).includes("«АЛМАЗ»"));
    assert.ok(protectedAnswerTokens(source).includes("«Договор подписан 21.10.2026»"));
});

test("language adaptation rejects altered legal numbers", async () => {
    const source={success:true,text:"Размер штрафа 1500 рублей."};
    const changed=await localizeResponse(source,"Describe the fine","en",
        async()=>"The fine is 15000 rubles.");
    assert.equal(changed.localizationStatus,"numeric-mismatch");
    assert.equal(changed.text,source.text);
});

test("keep Russian responses without spending another model request", async () => {
    const source={success:true,text:"Результат проверки: все необходимые условия выполнены, существенных ошибок не обнаружено."};
    const result=await localizeResponse(source,"Проверь результат","ru",
        async()=>{throw new Error("translator must not be called");});
    assert.equal(result.localizationStatus,"already-russian");
    assert.equal(result.text,source.text);
});

test("an explicit translation request is honored even when UI is Russian", async () => {
    const source={success:true,text:"Перевод: хорошая работа. Имя Jessica Core сохранено."};
    let called=false;
    const result=await localizeResponse(source,"Переведи на японский язык","ru",
        async(prompt)=>{called=true; assert.ok(prompt.includes("non-UI world languages")); return "良い仕事です。Jessica Core";});
    assert.equal(called,true);
    assert.equal(result.localizationStatus,"adapted");
    assert.ok(result.text.includes("Jessica Core"));
});
