import test from "node:test";
import assert from "node:assert/strict";
import {createDocumentEvidence} from "../core/files/documentEvidence.js";
import {validateAndRenderDocumentAnswer} from "../core/files/documentAnswerContract.js";
import {summarizeDocumentCitationCoverage} from "../core/files/documentCitationCoverage.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const answer = (ref, quote) => JSON.stringify({sections:[{heading:"Содержание",
    findings:[{text:"Зафиксирована позиция автора",kind:"author_claim",
        evidence:[{ref,quote}]}]}]});
test("verified citations are counted but never treated as whole-document review", async () => {
    const docs=[{name:"dissertation.docx",text:"Введение: автор формулирует проблему.\n\nГлава 1. Содержательная часть.",truncated:true}];
    const result=await analyzeDocumentAttachments({task:"Объясни позицию автора",attachments:docs},
        async () => answer("Ф1:Ф1","автор формулирует проблему"));
    assert.equal(result.success,true);
    assert.equal(result.citationCoverage.presentedFragments,2);
    assert.equal(result.citationCoverage.citedFragments,1);
    assert.equal(result.citationCoverage.extractionPartial,true);
    assert.equal(result.citationCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(result.citationCoverage.files[0].allPresentedFragmentsCited,false);
});
test("invalid and rejected quotations cannot increase citation counts", () => {
    const bundle=createDocumentEvidence([{name:"paper",text:"Автор указывает на научную проблему."}]);
    const checked=validateAndRenderDocumentAnswer(JSON.stringify({sections:[{
        heading:"Введение",findings:[
            {text:"Автор указывает на научную проблему",kind:"author_claim",
                evidence:[{ref:"Ф1:Ф1",quote:"Автор указывает на научную проблему"}]},
            {text:"Подтверждена новая правовая норма",kind:"source_fact",
                evidence:[{ref:"Ф1:Ф99",quote:"несуществующая норма"}]}
        ]}]}),bundle);
    assert.equal(checked.success,true);
    assert.deepEqual(checked.citedRefs,["Ф1:Ф1"]);
    assert.equal(summarizeDocumentCitationCoverage(bundle,checked.citedRefs).citedFragments,1);
    assert.equal(summarizeDocumentCitationCoverage(bundle,["Ф9:Ф3"]).citedFragments,0);
});
test("identically named files retain separate fragment coverage", () => {
    const bundle=createDocumentEvidence([
        {name:"part.txt",text:"Материал первого документа."},
        {name:"part.txt",text:"Материал второго документа."}
    ]);
    const coverage=summarizeDocumentCitationCoverage(bundle,["Ф2:Ф1"]);
    assert.equal(coverage.files.length,2);
    assert.equal(coverage.files[0].citedFragments,0);
    assert.equal(coverage.files[1].citedFragments,1);
    assert.equal(coverage.completeSourceAnalysisConfirmed,false);
});
test("all excerpt citations do not establish substantive full-file review", () => {
    const bundle=createDocumentEvidence([{name:"file",text:"Краткий материал."}]);
    const coverage=summarizeDocumentCitationCoverage(bundle,["Ф1:Ф1","Ф1:Ф1"]);
    assert.equal(coverage.citedFragments,1);
    assert.equal(coverage.files[0].allPresentedFragmentsCited,true);
    assert.equal(coverage.completeSourceAnalysisConfirmed,false);
});
