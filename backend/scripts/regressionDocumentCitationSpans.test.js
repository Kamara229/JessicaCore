import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {locateCitedDocumentFragments,scopeCitedDocumentFragments}
 from "../core/files/documentCitationSpans.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
const sha=value=>createHash("sha256").update(value,"utf8").digest("hex");
const pagesFor=(text,name)=>{
    const pages=[];let offset=0;
    do{const page=pageDocumentText(text,offset);pages.push({...page,name});offset=page.nextOffset;}
    while(offset!==null);
    return {name,pages,text:pages[0].text,truncated:text.length>16000};
};
const generate=async(_prompt,_task,docs)=>{
    const frag=createLosslessWindowEvidence(docs[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Анализ",findings:[{
        text:"Автор приводит положение",kind:"author_claim",
        evidence:[{ref:frag.ref,quote:frag.text.slice(0,Math.min(50,frag.text.length))}]
    }]}]});
};
test("citation offsets locate exact source excerpts without storing text",()=>{
    const text="Начало \n"+ "А".repeat(2400)+"🧪"+"Б".repeat(150);
    const bundle=createLosslessWindowEvidence({name:"x.txt",text});
    const window={number:4,sectionIndex:2,title:"Теория",startOffset:30000,
        endOffset:30000+text.length,text};
    const citedRefs=["Ф1:Ф2","Ф1:Ф1","Ф1:Ф2"];
    const spans=locateCitedDocumentFragments(window,bundle,{success:true,citedRefs});
    assert.equal(spans.length,2);
    for(const span of spans){
        const f=bundle.index.get(span.ref.replace(/^О4:/u,""));
        assert.equal(span.startOffset,window.startOffset+f.startOffset);
        assert.equal(span.endOffset,window.startOffset+f.endOffset);
        assert.equal(span.fragmentSha256,sha(f.text));
        assert.ok(!JSON.stringify(span).includes("А".repeat(100)));
    }
    assert.equal(scopeCitedDocumentFragments(spans,2)[0].ref,"Д2:О4:Ф1:Ф2");
});
test("untrusted citation or corrupt fragment cannot claim precise evidence coverage",()=>{
    const source="ВВЕДЕНИЕ\n"+"Правовое положение. ".repeat(150);
    const bundle=createLosslessWindowEvidence({name:"x",text:source});
    const window={number:1,sectionIndex:0,title:"Введение",startOffset:0,
        endOffset:source.length,text:source};
    assert.equal(locateCitedDocumentFragments(window,bundle,{success:true,citedRefs:["Ф1:Ф999"]}),null);
    assert.equal(locateCitedDocumentFragments(window,bundle,{success:true,citedRefs:[]}),null);
    assert.equal(locateCitedDocumentFragments({...window,text:"другой текст"},bundle,
        {success:true,citedRefs:["Ф1:Ф1"]}),null);
    assert.equal(scopeCitedDocumentFragments([{ref:"О1:Ф1:Ф1"}],4),null);
});
test("every cited span from multi-file source is bounded by that file text",async()=>{
    const a="ВВЕДЕНИЕ\n"+"Первое положение. ".repeat(1200)+"\nЗАКЛЮЧЕНИЕ\nВывод.";
    const b="ВВЕДЕНИЕ\n"+"Второе положение. ".repeat(1200)+"\nЗАКЛЮЧЕНИЕ\nВывод.";
    const files=[pagesFor(a,"same.docx"),pagesFor(b,"same.docx")];
    const result=await analyzeDocumentAttachments({task:"Проанализируй документы",attachments:files},generate);
    assert.equal(result.success,true);
    assert.ok(result.citationSpans.length>2);
    const seen=new Set();
    for(const item of result.citationSpans){
        const source=item.fileIndex===1?a:b;
        assert.ok(item.startOffset>=0);
        assert.ok(item.endOffset<=source.length);
        assert.ok(item.startOffset<item.endOffset);
        assert.equal(sha(source.slice(item.startOffset,item.endOffset)),item.fragmentSha256);
        assert.match(item.ref,/^Д[12]:О\d+:Ф1:Ф\d+$/u);
        seen.add(item.fileIndex);
    }
    assert.deepEqual([...seen].sort(),[1,2]);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
test("single paged file keeps window references and offsets",async()=>{
    const source="ВВЕДЕНИЕ\n"+"Историческое положение. ".repeat(1000);
    const result=await analyzeDocumentAttachments({task:"Прочитай текст",
        attachments:[pagesFor(source,"study.txt")]},generate);
    assert.equal(result.success,true);
    for(const s of result.citationSpans){
        assert.match(s.ref,/^О\d+:Ф1:Ф\d+$/u);
        assert.equal(sha(source.slice(s.startOffset,s.endOffset)),s.fragmentSha256);
    }
});
