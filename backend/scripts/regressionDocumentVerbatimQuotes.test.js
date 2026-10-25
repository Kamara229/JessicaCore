import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {validateAndRenderDocumentAnswer} from "../core/files/documentAnswerContract.js";
import {locateVerbatimQuoteRanges,scopeVerbatimQuoteRanges}
 from "../core/files/documentVerbatimQuotes.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const sha = text => createHash("sha256").update(text,"utf8").digest("hex");
const pages = (source,name)=>{
 const result=[];let at=0;
 do {const page=pageDocumentText(source,at);result.push({...page,name});at=page.nextOffset;}
 while(at!==null);
 return {name,pages:result};
};
const answer = (ref,quote)=>JSON.stringify({sections:[{heading:"Текст",findings:[{
    kind:"source_fact",text:"Проверяемое утверждение автора",
    evidence:[{ref,quote}]
}]}]});
const direct = async(_prompt,_task,docs)=>{
 const b=createLosslessWindowEvidence(docs[0]);
 const f=b.fragments[0];
 return answer(f.ref,f.text.slice(0,Math.min(75,f.text.length)));
};

test("verified answer retains its original quotes for precise source positioning",()=>{
 const text="ВВЕДЕНИЕ\nНаучное положение автора и опыт.";
 const b=createLosslessWindowEvidence({name:"x.txt",text});
 const result=validateAndRenderDocumentAnswer(answer("Ф1:Ф1","Научное положение автора"),b);
 assert.equal(result.success,true);
 assert.deepEqual(result.verifiedFindings[0].evidence,
     [{ref:"Ф1:Ф1",quote:"Научное положение автора"}]);
 const local={number:1,sectionIndex:0,startOffset:999,
     endOffset:999+text.length,text};
 const found=locateVerbatimQuoteRanges(local,b,result);
 assert.equal(found.exactLocatedQuotes,1);
 assert.equal(found.allVerbatimQuoteOffsetsLocated,true);
 assert.equal(found.ranges[0].startOffset,999+text.indexOf("Научное положение автора"));
 assert.equal(found.ranges[0].quoteSha256,sha("Научное положение автора"));
});

test("ambiguous repeated and normalized-but-not-verbatim quotes receive no invented offsets",()=>{
 const text="Исследование показывает результат. Исследование показывает результат.\nАвтор написал «закон».";
 const b=createLosslessWindowEvidence({name:"x.txt",text});
 const win={number:2,sectionIndex:0,startOffset:0,endOffset:text.length,text};
 const repeated=validateAndRenderDocumentAnswer(
     answer("Ф1:Ф1","Исследование показывает результат"),b);
 const ambiguous=locateVerbatimQuoteRanges(win,b,repeated);
 assert.equal(ambiguous.ambiguousQuotes,1);
 assert.equal(ambiguous.ranges.length,0);
 assert.equal(ambiguous.allVerbatimQuoteOffsetsLocated,false);
 const normal=validateAndRenderDocumentAnswer(answer("Ф1:Ф1",'Автор написал "закон"'),b);
 assert.equal(normal.success,true);
 const located=locateVerbatimQuoteRanges(win,b,normal);
 assert.equal(located.nonLiteralQuotes,1);
 assert.equal(located.ranges.length,0);
});

test("a forged source reference cannot obtain a quote offset",()=>{
 const text="ВВЕДЕНИЕ\nПравовая норма упоминается в источнике.";
 const b=createLosslessWindowEvidence({name:"x",text});
 const win={number:1,sectionIndex:0,startOffset:0,endOffset:text.length,text};
 const checked={success:true,citedRefs:["Ф1:Ф99"],verifiedFindings:[
     {evidence:[{ref:"Ф1:Ф99",quote:"Правовая норма"}]}
 ]};
 assert.equal(locateVerbatimQuoteRanges(win,b,checked),null);
 assert.equal(scopeVerbatimQuoteRanges([{ref:"wrong"}],2),null);
});

test("single complete document reports exact quoted character ranges independently of fragments",async()=>{
 const source="ВВЕДЕНИЕ\nУникальный тезис научного исследования и вывод.";
 const result=await analyzeDocumentAttachments({task:"Проанализируй",
     attachments:[pages(source,"paper.txt")]},direct);
 assert.equal(result.success,true);
 assert.equal(result.quoteCoverage.allVerbatimQuoteOffsetsLocated,true);
 assert.equal(result.quoteCoverage.evidenceQuotes,1);
 assert.equal(result.quoteRanges.length,1);
 const row=result.quoteRanges[0];
 assert.equal(row.quoteSha256,sha(source.slice(row.startOffset,row.endOffset)));
 assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});

test("two same-named documents retain separate exact quotation ranges",async()=>{
 const first="ВВЕДЕНИЕ\nУникальное мнение первого автора и доводы.";
 const second="ВВЕДЕНИЕ\nДругая уникальная позиция второго автора и доводы.";
 const result=await analyzeDocumentAttachments({task:"Сопоставь",attachments:[
    pages(first,"same.docx"),pages(second,"same.docx")
 ]},direct);
 assert.equal(result.success,true);
 assert.equal(result.quoteCoverage.allVerbatimQuoteOffsetsLocated,true);
 assert.deepEqual(result.quoteRanges.map(s=>s.fileIndex),[1,2]);
 assert.match(result.quoteRanges[0].ref,/^Д1:О1:Ф1:Ф1$/u);
 assert.match(result.quoteRanges[1].ref,/^Д2:О1:Ф1:Ф1$/u);
 assert.equal(sha(first.slice(result.quoteRanges[0].startOffset,
     result.quoteRanges[0].endOffset)),result.quoteRanges[0].quoteSha256);
 assert.equal(sha(second.slice(result.quoteRanges[1].startOffset,
     result.quoteRanges[1].endOffset)),result.quoteRanges[1].quoteSha256);
});
