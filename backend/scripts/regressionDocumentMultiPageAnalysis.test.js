import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {planMultiplePagedDocuments} from "../core/files/documentMultiPageAnalysis.js";
import {createHash} from "node:crypto";

const batch=(source,name)=>{
    const pages=[];let off=0;
    do{const page=pageDocumentText(source,off);pages.push({...page,name});off=page.nextOffset;}
    while(off!==null);
    return {name,text:pages[0].text,truncated:source.length>16000,pages};
};
const study1="ВВЕДЕНИЕ\n"+"Первое научное положение и обоснование. ".repeat(720)+"\nЗАКЛЮЧЕНИЕ\nВывод один.";
const study2="ВВЕДЕНИЕ\n"+"Второе научное положение и обоснование. ".repeat(720)+"\nЗАКЛЮЧЕНИЕ\nВывод два.";
const attachments=[batch(study1,"study.docx"),batch(study2,"study.docx")];
const mock=async(_prompt,_task,docs)=>{
    const evidence=createLosslessWindowEvidence(docs[0]);
    const first=evidence.fragments[0];
    const quote=first.text.slice(0,Math.min(60,first.text.length));
    return JSON.stringify({sections:[{heading:"Содержание",
        findings:[{text:"Автор приводит исследовательское положение",
            kind:"author_claim",evidence:[{ref:first.ref,quote}]}]}]});
};

test("two multi-page files use unique file+window citations with coverage for both",async()=>{
    const seen=[];
    const r=await analyzeDocumentAttachments({task:"Сравни научные положения",attachments},
        async(prompt,task,docs)=>{seen.push(docs[0].text);return mock(prompt,task,docs);});
    const plan=planMultiplePagedDocuments(attachments);
    assert.ok(plan);
    assert.equal(r.success,true);
    assert.equal(seen.join(""),study1+study2);
    assert.equal(r.readingCoverage.files.length,2);
    assert.equal(r.readingCoverage.processedWindows,plan.totalWindows);
    assert.equal(r.readingCoverage.files[0].sourceSha256,
        createHash("sha256").update(study1).digest("hex"));
    assert.equal(r.readingCoverage.files[1].sourceSha256,
        createHash("sha256").update(study2).digest("hex"));
    assert.match(r.text,/\[Д1:О1:Ф1:Ф1\]/);
    assert.match(r.text,/\[Д2:О1:Ф1:Ф1\]/);
    assert.match(r.text,/Документ 1: study.docx/);
    assert.match(r.text,/Документ 2: study.docx/);
    assert.equal(r.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.match(r.responseId,/^JDOC-/);
});

test("one corrupted file rejects the entire multi-file job before model calls",async()=>{
    const tampered={...attachments[1],pages:attachments[1].pages.map((p,i)=>
        i===1?{...p,text:"X"+p.text.slice(1)}:p)};
    let calls=0;
    const r=await analyzeDocumentAttachments({task:"Сравни",attachments:[attachments[0],tampered]},
        async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(r.success,false);
    assert.equal(r.stage,"document-multi-pages");
    assert.equal(r.readingCoverage.completeSourceAnalysisConfirmed,false);
});

test("mixed legacy and new page attachments cannot silently truncate a file",async()=>{
    let calls=0;
    const r=await analyzeDocumentAttachments({task:"Сравни",attachments:[
        attachments[0],{name:"legacy.docx",text:"a".repeat(16000),truncated:true}
    ]},async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(r.success,false);
    assert.equal(r.stage,"document-multi-pages");
});

test("bounded total model windows prevent runaway multi-file calls",async()=>{
    const huge="ВВЕДЕНИЕ\n"+"Правовое положение. ".repeat(12000);
    const big=batch(huge,"big.docx");
    assert.equal(planMultiplePagedDocuments([big,big]),null);
    let count=0;
    const r=await analyzeDocumentAttachments({task:"Проанализируй",attachments:[big,big]},
        async()=>{count++;return "{}";});
    assert.equal(count,0);
    assert.equal(r.success,false);
    assert.equal(r.stage,"document-multi-pages");
});

test("one unverified window makes the multi-document analysis incomplete",async()=>{
    let calls=0;
    const r=await analyzeDocumentAttachments({task:"Изучи",attachments},async(p,t,docs)=>{
        calls++;
        if(calls===3)return '{"sections":[]}';
        return mock(p,t,docs);
    });
    assert.equal(r.success,false);
    assert.equal(r.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.ok(r.readingCoverage.processedWindows<planMultiplePagedDocuments(attachments).totalWindows);
    assert.match(r.text,/не завершён/iu);
    assert.ok(r.readingCoverage.processedWindows>0);
    assert.equal(r.readingCoverage.incompleteFile.fileIndex,1);
    assert.equal(r.readingCoverage.incompleteFile.processedWindows,r.readingCoverage.processedWindows);
});
