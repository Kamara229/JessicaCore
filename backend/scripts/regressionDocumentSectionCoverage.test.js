import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {beginWindowLedger,noteVerifiedWindow} from "../core/files/documentWindowLedger.js";
import {auditDocumentSections} from "../core/files/documentSectionCoverage.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const original="ВВЕДЕНИЕ\n"+"Историческое обоснование. ".repeat(1500)+
  "\n1.1 Теоретические положения\n"+"Автор приводит аргументы. ".repeat(1500)+
  "\nЗАКЛЮЧЕНИЕ\nИтоговый вывод.";
const sourcePages=(text,name="study.docx")=>{
    const all=[];let offset=0;
    do{const page=pageDocumentText(text,offset);all.push({...page,name});offset=page.nextOffset;}
    while(offset!==null);
    return {name,pages:all};
};
const fakeModel=async(_prompt,_task,docs)=>{
    const item=createLosslessWindowEvidence(docs[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Выводы",findings:[{
        kind:"author_claim",text:"Автор приводит анализ ситуации",
        evidence:[{ref:item.ref,quote:item.text.slice(0,Math.min(65,item.text.length))}]
    }]}]});
};

test("each indexed section preserves exact spans and verified citations",async()=>{
    const response=await analyzeDocumentAttachments({task:"Проверь структуру",
        attachments:[sourcePages(original)]},fakeModel);
    assert.equal(response.success,true);
    const audit=response.sectionCoverage;
    assert.ok(audit.sectionCount>=3);
    assert.equal(audit.completedSections,audit.sectionCount);
    assert.equal(audit.allSectionsProcessed,true);
    assert.equal(audit.allSectionsCited,true);
    assert.equal(audit.semanticCorrectnessConfirmed,false);
    assert.equal(audit.sectionHeadingsHeuristic,true);
    assert.equal(audit.totalSourceChars,original.length);
    assert.equal(audit.sections.reduce((n,s)=>n+s.sourceChars,0),original.length);
    assert.equal(audit.sections[0].startOffset,0);
    assert.equal(audit.sections.at(-1).endOffset,original.length);
    for(const row of audit.sections){
        assert.ok(row.citedFragments>0);
        assert.ok(row.verifiedClaims>0);
        assert.equal(row.processedWindows,row.plannedWindows);
    }
});

test("partial processing records unprocessed sections and rejects full-status claims",()=>{
    const plan=planDocumentSectionWindows(original);
    const ledger=beginWindowLedger(plan,digest(original));
    assert.ok(ledger);
    assert.equal(noteVerifiedWindow(ledger,plan.windows[0],{
        success:true,kept:1,rejected:0,citedRefs:["Ф1:Ф1"]}),true);
    const audit=auditDocumentSections(plan,ledger,[],[]);
    assert.ok(audit);
    assert.equal(audit.allSectionsProcessed,false);
    assert.ok(audit.completedSections<audit.sectionCount);
    assert.equal(audit.allSectionsCited,false);
    assert.equal(audit.semanticCorrectnessConfirmed,false);
    assert.equal(audit.sections[0].processedWindows,1);
    assert.equal(audit.sections.at(-1).processedWindows,0);
});

test("source-offset or citation-hash manipulation invalidates the audit",()=>{
    const plan=planDocumentSectionWindows(original);
    const ledger=beginWindowLedger(plan,digest(original));
    assert.equal(auditDocumentSections({...plan,windows:plan.windows.map((w,i)=>
        i===0?{...w,endOffset:w.endOffset+1}:w)},ledger,[],[]),null);
    assert.equal(auditDocumentSections({...plan,windows:plan.windows.map((w,i)=>
        i===0?{...w,sectionIndex:3}:w)},ledger,[],[]),null);
    const forged={windowNumber:1,sectionIndex:0,startOffset:0,endOffset:10,
        ref:"О1:Ф1:Ф1",fragmentSha256:"0".repeat(64)};
    assert.equal(auditDocumentSections(plan,ledger,[forged],[]),null);
});

test("two same-name documents receive different per-file section maps",async()=>{
    const alternate="ВВЕДЕНИЕ\n"+"Другой научный материал. ".repeat(1150)+
        "\nЗАКЛЮЧЕНИЕ\nВторой вывод.";
    const response=await analyzeDocumentAttachments({
        task:"Сопоставь",attachments:[
            sourcePages(original,"same.docx"),sourcePages(alternate,"same.docx")]
    },fakeModel);
    assert.equal(response.success,true);
    const coverage=response.readingCoverage.files.map(x=>x.sectionCoverage);
    assert.equal(coverage[0].totalSourceChars,original.length);
    assert.equal(coverage[1].totalSourceChars,alternate.length);
    assert.equal(coverage[0].allSectionsProcessed,true);
    assert.equal(coverage[1].allSectionsProcessed,true);
    assert.equal(response.readingCoverage.completeSourceAnalysisConfirmed,false);
});
