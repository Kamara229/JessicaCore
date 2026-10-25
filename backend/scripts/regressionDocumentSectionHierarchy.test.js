import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {buildVerifiedSectionHierarchy}
 from "../core/files/documentSectionHierarchy.js";
import {beginWindowLedger,noteVerifiedWindow}
 from "../core/files/documentWindowLedger.js";
import {auditDocumentSections} from "../core/files/documentSectionCoverage.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const text="Преамбула без заголовка.\nВВЕДЕНИЕ\nЦель исследования.\n"+
    "ГЛАВА 1. Теоретические основы\nОбоснование.\n"+
    "1.1 Предмет исследования\nПозиции авторов.\n"+
    "1.1.1 Исторический контекст\nСудебная практика.\n"+
    "1.2 Практические следствия\nДругие аргументы.\n"+
    "ГЛАВА 2. Анализ\nНовые положения.\n"+
    "2.1 Сравнение\nПрактические данные.\n"+
    "ЗАКЛЮЧЕНИЕ\nКонечный вывод.";
const audit=(source=text)=>{
    const plan=planDocumentSectionWindows(source);
    const ledger=beginWindowLedger(plan,digest(source));
    return {plan,ledger};
};

test("root, chapter and nested paragraphs retain exact source offsets and parent paths",()=>{
    const {plan,ledger}=audit();
    assert.ok(plan.complete);
    const coverage=auditDocumentSections(plan,ledger,[],[]);
    assert.ok(coverage);
    const hierarchy=coverage.hierarchy;
    assert.equal(hierarchy.sections.length,9);
    assert.equal(hierarchy.headingHierarchyHeuristic,true);
    assert.equal(hierarchy.semanticStructureConfirmed,false);
    const [preface,intro,chapter,sub,subsub,sibling,chapter2,sub2,conclusion]=
        hierarchy.sections;
    assert.equal(preface.parentSectionIndex,null);
    assert.equal(intro.parentSectionIndex,null);
    assert.equal(chapter.parentSectionIndex,null);
    assert.equal(sub.parentSectionIndex,chapter.sectionIndex);
    assert.equal(subsub.parentSectionIndex,sub.sectionIndex);
    assert.equal(sibling.parentSectionIndex,chapter.sectionIndex);
    assert.equal(chapter2.parentSectionIndex,null);
    assert.equal(sub2.parentSectionIndex,chapter2.sectionIndex);
    assert.equal(conclusion.parentSectionIndex,null);
    assert.deepEqual(subsub.headingPath,[chapter.title,sub.title,subsub.title]);
    assert.deepEqual(sibling.headingPath,[chapter.title,sibling.title]);
    assert.equal(conclusion.depth,0);
    assert.equal(hierarchy.fullyProcessed,false);
    for(const row of hierarchy.sections)
        assert.equal(text.slice(row.startOffset,row.startOffset+row.title.length),row.title);
});

test("tampered heading offsets and titles cannot create a false hierarchy",()=>{
    const {plan,ledger}=audit();
    const coverage=auditDocumentSections(plan,ledger,[],[]);
    assert.ok(coverage);
    assert.equal(buildVerifiedSectionHierarchy(plan,coverage.sections.map((s,i)=>
        i===2?{...s,startOffset:s.startOffset+1}:s)),null);
    assert.equal(buildVerifiedSectionHierarchy(plan,coverage.sections.map((s,i)=>
        i===2?{...s,title:"Неизвестная глава"}:s)),null);
    assert.equal(buildVerifiedSectionHierarchy({...plan,sourceChars:plan.sourceChars-1},
        coverage.sections),null);
});

test("verified window progress remains linked to the corresponding nested heading",()=>{
    const {plan,ledger}=audit();
    const first=plan.windows[0];
    assert.equal(noteVerifiedWindow(ledger,first,{success:true,kept:1,
        rejected:0,citedRefs:["Ф1:Ф1"]}),true);
    const coverage=auditDocumentSections(plan,ledger,[],[]);
    const parent=coverage.hierarchy.sections.find(s=>s.sectionIndex===first.sectionIndex);
    assert.equal(parent.processedWindows,1);
    assert.ok(coverage.hierarchy.sections.some(s=>!s.complete));
    assert.equal(coverage.hierarchy.fullyProcessed,false);
});

test("document analysis returns section ancestry for source-backed citations",async()=>{
    const pages=[];let offset=0;
    do{const page=pageDocumentText(text,offset);pages.push({...page,name:"a.docx"});
       offset=page.nextOffset;}while(offset!==null);
    const answer=async(_,__,documents)=>{
        const part=createLosslessWindowEvidence(documents[0]).fragments[0];
        return JSON.stringify({sections:[{heading:"Источники",findings:[{
            kind:"author_claim",text:"Автор приводит положение",
            evidence:[{ref:part.ref,quote:part.text.slice(0,Math.min(30,part.text.length))}]
        }]}]});
    };
    const result=await analyzeDocumentAttachments({task:"Проанализируй структуру",
        attachments:[{name:"a.docx",pages}]},answer);
    assert.equal(result.success,true);
    assert.equal(result.sectionCoverage.hierarchy.fullyProcessed,true);
    assert.equal(result.sectionCoverage.hierarchy.sections.length,9);
    assert.equal(result.sectionCoverage.hierarchy.semanticStructureConfirmed,false);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
