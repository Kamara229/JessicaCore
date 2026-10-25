import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {measureDocumentCitationDensity}
 from "../core/files/documentEvidenceDensity.js";

const pages=(text,name)=>{
    const list=[];let offset=0;
    do{const item=pageDocumentText(text,offset);list.push({...item,name});offset=item.nextOffset;}
    while(offset!==null);
    return {name,pages:list};
};
const mk=async(_prompt,_task,attachments)=>{
    const bundle=createLosslessWindowEvidence(attachments[0]);
    const f=bundle.fragments[0];
    return JSON.stringify({sections:[{heading:"Фрагмент",findings:[{
        kind:"author_claim",text:"В тексте содержится положение",
        evidence:[{ref:f.ref,quote:f.text.slice(0,Math.min(f.text.length,45))}]
    }]}]});
};
test("submitted long documents distinguish exact citation anchors from read fragments",async()=>{
    const source="ВВЕДЕНИЕ\n"+"Подробный аргумент и контекст. ".repeat(1400)+
       "\nГЛАВА 1\n"+"Продолжение важного рассуждения. ".repeat(1600)+
       "\nЗАКЛЮЧЕНИЕ\nФинальный вывод.";
    const result=await analyzeDocumentAttachments({task:"Проанализируй",
       attachments:[pages(source,"research.docx")]},mk);
    assert.equal(result.success,true);
    const density=result.evidenceDensity;
    assert.equal(density.sourceFullySubmitted,true);
    assert.equal(density.submittedCharacters,source.length);
    assert.ok(density.fragments>density.fragmentsWithExactQuotes);
    assert.ok(density.fragmentsWithoutExactQuotes>0);
    assert.equal(density.semanticCompletenessEstablished,false);
    assert.ok(density.exactQuoteCharacters<source.length);
    assert.ok(density.sections.length>=2);
    assert.equal(density.sections.reduce((n,s)=>n+s.fragments,0),density.fragments);
});

test("duplicate and forged quotes cannot claim extra anchored fragments",()=>{
    const plan=planDocumentSectionWindows("ВВЕДЕНИЕ\n"+"Источник и цитата. ".repeat(100));
    const good=measureDocumentCitationDensity(plan,[]);
    assert.ok(good);
    assert.equal(good.fragmentsWithExactQuotes,0);
    assert.equal(good.fragmentsWithoutExactQuotes,good.fragments);
    assert.equal(measureDocumentCitationDensity(plan,[{
        ref:"О1:Ф1:Ф999",windowNumber:1,sectionIndex:0,startOffset:0,
        endOffset:5,quoteSha256:"f".repeat(64)
    }]),null);
});

test("identical filenames across different files preserve separate citation counts",async()=>{
    const a="ВВЕДЕНИЕ\n"+"Первый документ содержит факты. ".repeat(900);
    const b="ВВЕДЕНИЕ\n"+"Второй документ содержит факты. ".repeat(1000);
    const result=await analyzeDocumentAttachments({task:"Сопоставь",
        attachments:[pages(a,"same.docx"),pages(b,"same.docx")]},mk);
    assert.equal(result.success,true);
    const coverage=result.evidenceDensity;
    assert.equal(coverage.files.length,2);
    assert.equal(coverage.files[0].submittedCharacters,a.length);
    assert.equal(coverage.files[1].submittedCharacters,b.length);
    assert.equal(coverage.submittedCharacters,a.length+b.length);
    assert.equal(coverage.semanticCompletenessEstablished,false);
    assert.ok(coverage.fragmentsWithoutExactQuotes>0);
    assert.equal(coverage.files[0].fileIndex,1);
    assert.equal(coverage.files[1].fileIndex,2);
});

test("overlapping citations count each quoted source character once",()=>{
    const source="ВВЕДЕНИЕ\n1234567890abcdefghijklmnopqrstuvwxyz";
    const plan=planDocumentSectionWindows(source);
    const quote=(start,end)=>({
        ref:"О1:Ф1:Ф1",windowNumber:1,sectionIndex:0,
        startOffset:start,endOffset:end,
        quoteSha256:requireDigest(source.slice(start,end))
    });
    const report=measureDocumentCitationDensity(plan,[quote(9,19),quote(14,24)]);
    assert.ok(report);
    assert.equal(report.exactQuoteCharacters,15);
    assert.equal(report.fragmentsWithExactQuotes,1);
});
import {createHash} from "node:crypto";
function requireDigest(value){return createHash("sha256").update(value,"utf8").digest("hex");}
