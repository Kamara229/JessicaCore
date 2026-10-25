import test from "node:test";
import assert from "node:assert/strict";
import {createVerifiedClaimRecords,scopeVerifiedClaimRecords,compareVerifiedDocumentClaims}
 from "../core/files/documentClaimComparison.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const docs=(s,name)=>{
  const pages=[];let offset=0;
  do{const page=pageDocumentText(s,offset);pages.push({...page,name});offset=page.nextOffset;}while(offset!==null);
  return {name,pages};
};
const model=async(_prompt,_task,items)=>{
  const ref=createLosslessWindowEvidence(items[0]).fragments[0];
  return JSON.stringify({sections:[{heading:"Позиция",findings:[{
    kind:"author_claim",text:"Автор указывает на необходимость дополнительной проверки",
    evidence:[{ref:ref.ref,quote:ref.text.slice(0,Math.min(55,ref.text.length))}]
  }]}]});
};
test("only validated local claims with cited spans enter the comparison",()=>{
 const findings=[{text:"Тезис",kind:"author_claim",heading:"Тема",refs:["Ф1:Ф1"]}];
 const spans=[{ref:"О2:Ф1:Ф1"}];
 const records=createVerifiedClaimRecords(findings,spans,2);
 assert.equal(records[0].refs[0],"О2:Ф1:Ф1");
 assert.equal(createVerifiedClaimRecords([{...findings[0],refs:["Ф1:Ф9"]}],spans,2),null);
 assert.equal(createVerifiedClaimRecords([{...findings[0],kind:"fiction"}],spans,2),null);
 assert.equal(scopeVerifiedClaimRecords(records,2,[{ref:"Д2:О2:Ф1:Ф1"}])[0].refs[0],
    "Д2:О2:Ф1:Ф1");
 assert.equal(scopeVerifiedClaimRecords(records,2,[{ref:"Д1:О2:Ф1:Ф1"}]),null);
});

test("identical claims from different chapters are grouped without truth assertions",()=>{
 const source="ВВЕДЕНИЕ\n"+"Правовая позиция автора. ".repeat(1200)+
   "\nГЛАВА 1. ПРАВОВАЯ ПРИРОДА\n"+"Иное научное положение. ".repeat(1200);
 return analyzeDocumentAttachments({task:"Сопоставь тезисы",
    attachments:[docs(source,"dissertation.docx")]},model).then(r=>{
   assert.equal(r.success,true);
   assert.equal(r.claimComparison.totalVerifiedClaims,r.groundedClaims);
   assert.ok(r.claimComparison.repeatedFormulations>=1);
   assert.equal(r.claimComparison.crossFileExactRepeats,0);
   assert.equal(r.claimComparison.semanticContradictionsAssessed,false);
   assert.equal(r.claimComparison.independentCorroborationEstablished,false);
 });
});

test("separate file identities survive matching filenames and repeated statements",async()=>{
 const a="ВВЕДЕНИЕ\n"+"Содержание исследования А. ".repeat(950);
 const b="ВВЕДЕНИЕ\n"+"Содержание исследования Б. ".repeat(950);
 const r=await analyzeDocumentAttachments({task:"Сопоставь два документа",
    attachments:[docs(a,"same.docx"),docs(b,"same.docx")]},model);
 assert.equal(r.success,true);
 assert.equal(r.claimComparison.totalVerifiedClaims,r.groundedClaims);
 assert.ok(r.claimComparison.crossFileExactRepeats>0);
 const evidence=r.claimComparison.examples.find(x=>
   new Set(x.occurrences.map(o=>o.fileIndex)).size>1);
 assert.ok(evidence);
 assert.ok(evidence.occurrences.some(x=>x.refs[0].startsWith("Д1:")));
 assert.ok(evidence.occurrences.some(x=>x.refs[0].startsWith("Д2:")));
 assert.equal(r.claimComparison.independentCorroborationEstablished,false);
});

test("similar or opposing meanings are not automatically called contradictions",()=>{
 const spans=[{ref:"О1:Ф1:Ф1"}];
 const a={text:"Обязательство возникает",kind:"source_fact",
   refs:["О1:Ф1:Ф1"],fileIndex:1,windowNumber:1};
 const b={...a,text:"Обязательство не возникает"};
 const summary=compareVerifiedDocumentClaims([a,b],spans);
 assert.equal(summary.repeatedFormulations,0);
 assert.equal(summary.semanticContradictionsAssessed,false);
 assert.equal(compareVerifiedDocumentClaims([{...a,refs:["О2:Ф1:Ф1"]}],spans),null);
});

test("repeat normalization ignores only spacing/case; classification is preserved",()=>{
 const spans=[{ref:"О1:Ф1:Ф1"},{ref:"О2:Ф1:Ф1"}];
 const a={text:"Вывод   автора",kind:"author_claim",refs:["О1:Ф1:Ф1"],fileIndex:1,windowNumber:1};
 const b={...a,text:"вывод автора",refs:["О2:Ф1:Ф1"],windowNumber:2};
 const summary=compareVerifiedDocumentClaims([a,b],spans);
 assert.equal(summary.repeatedFormulations,1);
 assert.equal(compareVerifiedDocumentClaims([a,{...b,kind:"source_fact"}],spans).repeatedFormulations,0);
});
