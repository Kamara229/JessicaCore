import test from "node:test";
import assert from "node:assert/strict";
import {buildDocumentPageChunk} from "../core/files/documentPageChunk.js";
import {assembleDocumentChunks} from "../core/files/documentChunkAssembly.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";

const source="ВВЕДЕНИЕ\n"+"Юридический аргумент и проверяемые положения. ".repeat(8300)+
    "\nГЛАВА 1. Проблема\n"+"Сопоставление правовых норм. ".repeat(1700);
const allChunks=(text,name="thesis.docx")=>{
    const chunks=[];
    let offset=0,hash=null;
    do{
        const item=buildDocumentPageChunk(text,name,offset,hash);
        if(!item)throw Error("Chunk extraction rejected");
        chunks.push(item);
        offset=item.nextOffset;
        hash=item.sourceSha256;
        if(chunks.length>16)throw Error("Unexpected chunk count");
    }while(offset!==null);
    return chunks;
};
const model=async(_prompt,_task,excerpts)=>{
    const frag=createLosslessWindowEvidence(excerpts[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Проверенный материал",findings:[{
        kind:"author_claim",text:"Автор приводит исследуемое положение",
        evidence:[{ref:frag.ref,quote:frag.text.slice(0,Math.min(frag.text.length,40))}]
    }]}]});
};

test("complete chunk series proves original source SHA and exact text",()=>{
    const chunks=allChunks(source);
    assert.ok(chunks.length>=2);
    const result=assembleDocumentChunks(chunks);
    assert.equal(result.complete,true);
    assert.equal(result.sourceIntegrityVerified,true);
    assert.equal(result.sourceSha256,chunks[0].sourceSha256);
    assert.equal(result.pageCount,chunks.reduce((n,x)=>n+x.pageCount,0));
    assert.equal(result.text,source);
    assert.equal(result.modelAnalysisPerformed,false);
    assert.equal(result.semanticCompletenessConfirmed,false);
});
test("missing final chunk is explicitly incomplete and cannot be submitted as a document",async()=>{
    const chunks=allChunks(source).slice(0,1);
    const partial=assembleDocumentChunks(chunks);
    assert.equal(partial.complete,false);
    assert.equal(partial.sourceIntegrityVerified,false);
    assert.ok(partial.nextOffset>0);
    assert.equal(Object.hasOwn(partial,"text"),false);
    assert.equal(Object.hasOwn(partial,"pages"),false);
    let calls=0;
    const analyzed=await analyzeDocumentAttachments({task:"Прочитай",
        attachments:[{name:"thesis.docx",chunks}]},async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(analyzed.success,false);
    assert.equal(analyzed.stage,"document-chunks-incomplete");
    assert.equal(analyzed.readingCoverage.complete,false);
});
test("gap, duplicate, swapped, cross-document and altered metadata fail closed",()=>{
    const chunks=allChunks(source);
    const first=chunks[0],second=chunks[1];
    assert.equal(assembleDocumentChunks([second,first]),null);
    assert.equal(assembleDocumentChunks([first,first]),null);
    assert.equal(assembleDocumentChunks([first,{...second,startOffset:second.startOffset+1}]),null);
    assert.equal(assembleDocumentChunks([first,{...second,sourceSha256:"0".repeat(64)}]),null);
    assert.equal(assembleDocumentChunks([first,{...second,pages:second.pages.map((p,i)=>
        i===0?{...p,text:"X"+p.text.slice(1)}:p)}]),null);
    assert.equal(assembleDocumentChunks([first,{...second,pages:second.pages.map((p,i)=>
        i===0?{...p,name:"other.docx"}:p)}]),null);
    assert.equal(assembleDocumentChunks([first,{...second,sourceFullyCollected:true}]),null);
    assert.equal(assembleDocumentChunks([first,{...second,modelAnalysisPerformed:true}]),null);
});
test("small chunked file reaches existing verified analysis path",async()=>{
    const small="ВВЕДЕНИЕ\n"+"Исходный аргумент. ".repeat(3000)+
        "\nЗАКЛЮЧЕНИЕ\nЗаключительная позиция.";
    const chunks=allChunks(small,"small.docx");
    const result=await analyzeDocumentAttachments({task:"Проверь текст",
        attachments:[{name:"small.docx",chunks}]},model);
    assert.equal(result.success,true);
    assert.equal(result.readingCoverage.allWindowsPresented,true);
    assert.equal(result.readingCoverage.sourceSha256,chunks[0].sourceSha256);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
test("oversized chunked document is still refused by AI planning instead of silent truncation",async()=>{
    const chunks=allChunks(source);
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Проанализируй полностью",
        attachments:[{name:"thesis.docx",chunks}]},async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(result.success,false);
    assert.equal(result.stage,"document-read-plan");
    assert.equal(result.readingCoverage.processedWindows,0);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
