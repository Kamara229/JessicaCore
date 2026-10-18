import test from "node:test";
import assert from "node:assert/strict";
import { buildDocumentPrompt, analyzeDocumentAttachments } from "../core/files/documentAnalysisFacade.js";
import { createDocumentEvidence } from "../core/files/documentEvidence.js";
import {
    documentAnalysisInstructions,
    validateAndRenderDocumentAnswer
} from "../core/files/documentAnswerContract.js";

const file = {
    name: "О замене конструктива потолков.docx",
    text: "Акт согласования потолков. В проекте заменяется подвесная конструкция потолка. Работы согласуются сторонами.",
    truncated: false
};
const verified = (text = "В проекте предлагается заменить подвесной потолок") => JSON.stringify({
    sections: [{
        heading: "Основные положения",
        findings: [{
            text,
            kind: "author_claim",
            evidence: [{
                ref: "Ф1:Ф1",
                quote: "В проекте заменяется подвесная конструкция потолка"
            }]
        }]
    }]
});

test("document contents reaches model with fragment identifiers and quotations", async () => {
    let calls = 0;
    const result = await analyzeDocumentAttachments({
        task: "Проанализируй документ и расскажи о его содержании.",
        attachments: [file]
    }, async prompt => {
        calls++;
        assert.match(prompt, /В проекте заменяется подвесная конструкция/);
        assert.match(prompt, /О замене конструктива потолков\.docx/);
        assert.match(prompt, /\[Ф1:Ф1\]/);
        return verified();
    });
    assert.equal(calls, 1);
    assert.equal(result.success, true);
    assert.equal(result.evidenceChecked, true);
    assert.equal(result.validated, false); // quotes checked ≠ legal truth verified
    assert.match(result.text, /Позиция автора/);
    assert.match(result.text, /Основание: «В проекте заменяется/);
    assert.match(result.text, /\[Ф1:Ф1\]/);
    assert.match(result.text, /не юридическую правильность/);
});

test("file excerpt caps and partial-document warning are preserved", () => {
    const prompt = buildDocumentPrompt({
        task: "Изучи",
        attachments: [{name: "large.txt", text: "A".repeat(25000), truncated: true}]
    });
    assert.match(prompt, /обрезана/);
    assert.match(prompt, /\[Ф1:Ф1\]/);
    assert.ok(prompt.length < 18500);
});

test("two documents get independent, stable citations", () => {
    const bundle = createDocumentEvidence({
        length: 0
    });
    assert.equal(bundle.docs.length, 0);
    const two = createDocumentEvidence([
        {name:"a.txt", text:"Первый документ сообщает: соглашение подписано."},
        {name:"b.txt", text:"Второй документ сообщает: проект отклонён."}
    ]);
    assert.ok(two.index.has("Ф1:Ф1"));
    assert.ok(two.index.has("Ф2:Ф1"));
    assert.match(two.promptText, /Второй документ сообщает/);
});

test("fabricated quote or non-existent source ref is always removed", () => {
    const bundle = createDocumentEvidence([file]);
    const attempted = JSON.stringify({
        sections: [{heading:"Пункты",findings:[
            {
                text:"В проекте предусмотрено устройство бассейна",
                kind:"source_fact",
                evidence:[{ref:"Ф1:Ф1",quote:"в проекте есть бассейн"}]
            },
            {
                text:"Предусмотрено устройство бассейна",
                kind:"source_fact",
                evidence:[{ref:"Ф2:Ф10",quote:"В проекте заменяется подвесная конструкция потолка"}]
            }
        ]}]
    });
    const result = validateAndRenderDocumentAnswer(attempted, bundle);
    assert.equal(result.success, false);
    assert.equal(result.kept, 0);
    assert.equal(result.rejected, 2);
});

test("unverified claims are dropped while supported facts survive", () => {
    const bundle = createDocumentEvidence([file]);
    const payload = JSON.stringify({
        sections:[{heading:"Основное",findings:[
            {text:"В проекте есть потолок",kind:"source_fact",
             evidence:[{ref:"Ф1:Ф1",quote:"заменяется подвесная конструкция потолка"}]},
            {text:"В проекте указана стоимость 700000 рублей",kind:"source_fact",
             evidence:[{ref:"Ф1:Ф1",quote:"заменяется подвесная конструкция потолка"}]}
        ]}]
    });
    const result = validateAndRenderDocumentAnswer(payload, bundle);
    assert.equal(result.success, true);
    assert.equal(result.kept, 1);
    assert.equal(result.rejected, 1);
    assert.match(result.text, /Не включены 1 пункт/);
    assert.doesNotMatch(result.text, /700000/);
});

test("legal claims are attributed to document authors, not independent law", () => {
    const note = documentAnalysisInstructions();
    assert.match(note, /не считай авторские утверждения/i);
    assert.match(note, /юридическая/i);
    const rendered = validateAndRenderDocumentAnswer(verified(
        "Автор письма предлагает заменить потолок; законность предложения отдельно не подтверждена"
    ), createDocumentEvidence([file]));
    assert.equal(rendered.success, true);
    assert.match(rendered.text, /Позиция автора/);
    assert.match(rendered.text, /нормы и независимые источники отдельно не проверялись/);
});

test("truncated documents carry an explicit limitation", () => {
    const bundle = createDocumentEvidence([{
        ...file,
        truncated:true
    }]);
    const result = validateAndRenderDocumentAnswer(verified(), bundle);
    assert.equal(result.success,true);
    assert.match(result.text, /доступны частично/);
});

test("empty or invalid documents never call the model", async () => {
    const result = await analyzeDocumentAttachments({
        task:"Изучи файл", attachments:[{name:"empty",text:""}]
    }, async()=>{throw Error("unexpected");});
    assert.equal(result.success,false);
    assert.match(result.text,/не содержит доступного текста/);
});

test("non-JSON model response is rejected, not silently passed through", async () => {
    const result = await analyzeDocumentAttachments({
        task:"Перескажи",attachments:[file]
    }, async()=> "В этом документе точно есть совершенно новая информация");
    assert.equal(result.success,false);
    assert.equal(result.stage, "document-evidence");
});

test("provider failure must never fabricate analysis", async () => {
    const result = await analyzeDocumentAttachments(
        {task:"Перескажи",attachments:[file]},
        async()=>{const error=new Error("bad");error.status=503;throw error;}
    );
    assert.equal(result.success,false);
    assert.match(result.text,/Не удалось завершить анализ/);
});

test("numeric evidence uses exact tokens rather than matching number substrings", () => {
    const bundle = createDocumentEvidence([{
        name:"quotation.txt",
        text:"Стоимость монтажа составляет 15000 рублей, без учёта материалов."
    }]);
    const bad = JSON.stringify({sections:[{heading:"Стоимость",findings:[
        {text:"Стоимость монтажа составляет 5000 рублей",kind:"source_fact",
         evidence:[{ref:"Ф1:Ф1",quote:"Стоимость монтажа составляет 15000 рублей"}]}
    ]}]});
    const result=validateAndRenderDocumentAnswer(bad,bundle);
    assert.equal(result.success,false);
    assert.equal(result.rejected,1);
});

test("a document author's standards-compliance claim is not established legal truth", () => {
    const bundle = createDocumentEvidence([{
        name:"письмо.txt",
        text:"По мнению автора, потолок CLIP не соответствует требованиям СП 2.13130.2020."
    }]);
    const payload=JSON.stringify({sections:[{heading:"Пожарная безопасность",findings:[
        {text:"Потолок CLIP не соответствует требованиям СП 2.13130.2020",kind:"source_fact",
         evidence:[{ref:"Ф1:Ф1",quote:"потолок CLIP не соответствует требованиям СП 2.13130.2020"}]}
    ]}]});
    const result=validateAndRenderDocumentAnswer(payload,bundle);
    assert.equal(result.success,true);
    assert.match(result.text,/Позиция автора: Потолок CLIP/);
    assert.doesNotMatch(result.text,/В документе указано: Потолок CLIP/);
    assert.match(result.text,/независимые источники отдельно не проверялись/);
});

test("broad document summaries prioritize actual proposed changes over headers", () => {
    const source = createDocumentEvidence([{
        name: "technical-letter.txt",
        text: [
            "Название документа: О состоянии строительства.",
            "Контактные данные: телефон и факс.",
            "Детали: предлагается изменить потолок с целью экономии средств.",
            "Стоимость работ составляет 500000 рублей.",
            "Вспомогательные помещения также предлагается изменить."
        ].join("\n\n")
    }]);
    const input = JSON.stringify({
        sections: [{heading:"Содержание", findings:[
            {text:"Название документа: О состоянии строительства",kind:"source_fact",evidence:[{ref:"Ф1:Ф1",quote:"Название документа: О состоянии строительства"}]},
            {text:"Указаны контактные данные: телефон и факс",kind:"source_fact",evidence:[{ref:"Ф1:Ф2",quote:"Контактные данные: телефон и факс"}]},
            {text:"Предлагается изменить потолок с целью экономии средств",kind:"author_claim",evidence:[{ref:"Ф1:Ф3",quote:"предлагается изменить потолок с целью экономии средств"}]},
            {text:"Стоимость работ составляет 500000 рублей",kind:"source_fact",evidence:[{ref:"Ф1:Ф4",quote:"Стоимость работ составляет 500000 рублей"}]},
            {text:"Вспомогательные помещения также предлагается изменить",kind:"author_claim",evidence:[{ref:"Ф1:Ф5",quote:"Вспомогательные помещения также предлагается изменить"}]}
        ]}]
    });
    const summary = validateAndRenderDocumentAnswer(input, source, "Перескажи основные положения документа");
    assert.equal(summary.success, true);
    assert.equal(summary.kept, 3);
    assert.equal(summary.omittedAsSecondary, 2);
    assert.doesNotMatch(summary.text, /Указаны контактные данные/);
    assert.match(summary.text, /Вспомогательные помещения также предлагается изменить/);
    const contacts = validateAndRenderDocumentAnswer(input, source, "Укажи контакты, название и отправителя");
    assert.equal(contacts.kept, 5);
    assert.match(contacts.text, /контактные данные/);
});

test("a numerical claim cannot cite a different quotation within the same paragraph", () => {
    const source = createDocumentEvidence([{
        name:"expenses.txt",
        text:"В письме приведена сумма 500000 рублей. Также описано согласование конструкции."
    }]);
    const output = JSON.stringify({sections:[{heading:"Расходы",findings:[{
        text:"Стоимость конструкции составляет 500000 рублей",
        kind:"source_fact",
        evidence:[{ref:"Ф1:Ф1",quote:"Также описано согласование конструкции"}]
    }]}]});
    const result = validateAndRenderDocumentAnswer(output, source);
    assert.equal(result.success, false);
    assert.equal(result.rejected, 1);
});

test("each generated document response carries a matching trace identity", async () => {
    const success = await analyzeDocumentAttachments(
        {task:"Перескажи документ",attachments:[file]},
        async()=>verified()
    );
    assert.equal(success.success,true);
    assert.match(success.responseId,/^JDOC-[A-F0-9]{10}$/);
    assert.ok(success.responseSha256?.length === 64);
    assert.match(success.text, /Код ответа: JDOC-[A-F0-9]{10}/);
    assert.ok(success.text.includes(success.responseId));
    const { createHash } = await import("node:crypto");
    assert.equal(success.responseSha256, createHash("sha256").update(success.text).digest("hex"));
});
