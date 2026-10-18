import test from "node:test";
import assert from "node:assert/strict";
import { buildDocumentPrompt, analyzeDocumentAttachments } from "../core/files/documentAnalysisFacade.js";

const file = {name:"О замене конструктива потолков.docx",text:"Акт согласования потолков. В проекте заменяется подвесная конструкция потолка. Работы согласуются сторонами.",truncated:false};
test("document analysis reads provided document, not a web search", async () => {
    let called = 0;
    const result = await analyzeDocumentAttachments({
        task:"Проанализируй приложенный файл и расскажи об его содержании.",
        attachments:[file]
    }, async prompt => {
        called++;
        assert.match(prompt,/В проекте заменяется подвесная конструкция/);
        assert.match(prompt,/О замене конструктива потолков\.docx/);
        return "В документе рассматривается замена конструкции потолка.";
    });
    assert.equal(called,1);
    assert.equal(result.success,true);
    assert.equal(result.mode,"document");
    assert.match(result.text,/замена конструкции потолка/);
});
test("document context cap and truncated flag are stated to model", () => {
    const prompt = buildDocumentPrompt({task:"Изучи",attachments:[
        {name:"large.txt",text:"A".repeat(25000),truncated:true}
    ]});
    assert.match(prompt,/ФРАГМЕНТ/);
    assert.ok(prompt.length < 18000);
});
test("invalid or empty document fails clearly, without web search", async () => {
    const result = await analyzeDocumentAttachments({task:"Изучи файл",attachments:[{name:"empty",text:""}]}, async()=> {throw Error("unexpected");});
    assert.equal(result.success,false);
    assert.match(result.text,/не содержит доступного текста/);
});
test("provider failure must not fabricate analysis", async () => {
    const result = await analyzeDocumentAttachments(
        {task:"Перескажи",attachments:[file]},
        async()=> {const error=new Error("bad");error.status=503;throw error;}
    );
    assert.equal(result.success,false);
    assert.match(result.text,/Не удалось завершить анализ/);
});
