import test from "node:test";
import assert from "node:assert/strict";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

test("lossless window preserves trailing spaces, blank lines and Unicode",()=>{
    const text="ВВЕДЕНИЕ\n"+"  абзац\n\n".repeat(1200)+"🧪  ";
    const data=createLosslessWindowEvidence({name:"research.docx",text});
    assert.ok(data?.exactSourceCoverage);
    assert.equal(data.partial,false);
    assert.equal(data.omitted,0);
    assert.equal(data.fragments.map(f=>f.text).join(""),text);
    assert.equal(data.fragments[0].startOffset,0);
    assert.equal(data.fragments.at(-1).endOffset,text.length);
    for(const frag of data.fragments){
        assert.equal(frag.text,text.slice(frag.startOffset,frag.endOffset));
        assert.equal(data.index.get(frag.ref),frag);
        assert.ok(frag.text.length<=900);
    }
});
test("a thousand short paragraphs are all retained instead of dropping after 100",()=>{
    const text="Пункт.\n\n".repeat(1400);
    const data=createLosslessWindowEvidence({name:"article.txt",text});
    assert.equal(data.fragments.map(f=>f.text).join(""),text);
    assert.ok(data.fragments.length<100);
    assert.equal(data.exactSourceCoverage,true);
});
test("surrogate pair at fragment boundary remains intact",()=>{
    const text="А".repeat(899)+"🧪"+"Б".repeat(900)+"🧬";
    const data=createLosslessWindowEvidence({name:"file",text});
    assert.equal(data.fragments.map(f=>f.text).join(""),text);
    for(const frag of data.fragments){
        assert.ok(!/[\uD800-\uDBFF]$/u.test(frag.text));
        assert.ok(!/^[\uDC00-\uDFFF]/u.test(frag.text));
    }
});
test("paged document analysis uses complete first window evidence for short paragraphs",async()=>{
    const text="ВВЕДЕНИЕ\n"+"Указан факт.\n\n".repeat(700);
    const pages=[];let at=0;
    do{
        const p=pageDocumentText(text,at);
        pages.push({...p,name:"research.docx"});at=p.nextOffset;
    }while(at!==null);
    const seen=[];
    const result=await analyzeDocumentAttachments({
        task:"Проверь выводы",attachments:[{name:"research.docx",pages}]
    },async (prompt,_task,files)=>{
        seen.push(files[0].text);
        const evidence=createLosslessWindowEvidence({name:files[0].name,text:files[0].text});
        const frag=evidence.fragments[0];
        assert.ok(frag.text.includes("Указан факт"));
        return JSON.stringify({sections:[{heading:"Анализ",findings:[{
            text:"В тексте приведён факт",kind:"source_fact",
            evidence:[{ref:frag.ref,quote:"Указан факт"}]
        }]}]});
    });
    assert.equal(result.success,true);
    assert.equal(seen.join(""),text);
    assert.equal(result.readingCoverage.exactTextCoverageConfirmed,true);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
test("empty and oversized inputs are rejected",()=>{
    assert.equal(createLosslessWindowEvidence({name:"x",text:""}),null);
    assert.equal(createLosslessWindowEvidence({name:"x",text:"a".repeat(16001)}),null);
});
