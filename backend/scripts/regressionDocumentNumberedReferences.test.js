import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {auditDocumentNumberedReferences} from "../core/files/documentNumberedReferenceAudit.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const academic="ВВЕДЕНИЕ\nПоложение изложено [1, с. 28].\n"+
    "ГЛАВА 1. Теория\nРазногласия [2, с. 15; 3, с. 48].\n"+
    "1.1 Содержание\nДругое мнение [2, с. 18].\n"+
    "ЗАКЛЮЧЕНИЕ\nВывод [4].\n"+
    "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\n"+
    "1. Автор. Книга.\n2. Второй автор. Статья.\n"+
    "2. Повторный источник.\n5. Исследование.\n";
const pages=(source,name)=>{
    const result=[];let at=0;
    do{const item=pageDocumentText(source,at);result.push({...item,name});at=item.nextOffset;}
    while(at!==null);
    return {name,pages:result};
};
const model=async(_prompt,_task,docs)=>{
    const first=createLosslessWindowEvidence(docs[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Выводы",findings:[{
        kind:"author_claim",text:"Автор рассматривает тему исследования",
        evidence:[{ref:first.ref,quote:first.text.slice(0,Math.min(first.text.length,45))}]
    }]}]});
};
test("numbered academic references count ranges and preserve original source offsets",()=>{
    const report=auditDocumentNumberedReferences(academic,digest(academic));
    assert.ok(report);
    assert.equal(report.bibliographyHeadingFound,true);
    assert.equal(report.citationMentions,5);
    assert.equal(report.bibliographyEntries,4);
    assert.deepEqual(report.missingEntryNumbers,[3,4]);
    assert.deepEqual(report.unreferencedEntryNumbers,[5]);
    assert.deepEqual(report.duplicateEntryNumbers,[2]);
    assert.equal(report.sourceRelevanceVerified,false);
    for(const ref of report.inTextReferences){
        assert.equal(ref.literalSha256,digest(academic.slice(ref.startOffset,ref.endOffset)));
        assert.ok(ref.endOffset<=academic.indexOf("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ"));
    }
});
test("no bibliography does not invent unsupported missing-reference verdicts",()=>{
    const source="ВВЕДЕНИЕ\nФормула [1] и условие [2, c. 3].\n";
    const report=auditDocumentNumberedReferences(source,digest(source));
    assert.equal(report.bibliographyHeadingFound,false);
    assert.equal(report.citationMentions,2);
    assert.deepEqual(report.missingEntryNumbers,[]);
    assert.equal(report.bibliographyCompletenessConfirmed,false);
});
test("source hash mismatch fails closed and arbitrary bracket content is not accepted",()=>{
    assert.equal(auditDocumentNumberedReferences(academic,"0".repeat(64)),null);
    const source="ВВЕДЕНИЕ\nНастройка [debug] и [1, 2] не ссылки по строгому шаблону.";
    assert.equal(auditDocumentNumberedReferences(source,digest(source)).citationMentions,0);
});
test("normal single and multi document analysis return independent numbered references",async()=>{
    const first=await analyzeDocumentAttachments({task:"Анализ диссертации",
        attachments:[pages(academic,"thesis.docx")]},model);
    assert.equal(first.success,true);
    assert.deepEqual(first.referenceAudit.missingEntryNumbers,[3,4]);
    const other="ВВЕДЕНИЕ\nТезис [8].\n"+
        "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\n8. Автор. Труд.\n";
    const combined=await analyzeDocumentAttachments({task:"Сопоставь",
        attachments:[pages(academic,"same.docx"),pages(other,"same.docx")]},model);
    assert.equal(combined.success,true);
    assert.equal(combined.referenceAudits.length,2);
    assert.deepEqual(combined.referenceAudits[0].missingEntryNumbers,[3,4]);
    assert.deepEqual(combined.referenceAudits[1].missingEntryNumbers,[]);
    assert.equal(combined.referenceAudits[0].fileIndex,1);
    assert.equal(combined.referenceAudits[1].fileIndex,2);
});
