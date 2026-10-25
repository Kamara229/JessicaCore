import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {beginWindowLedger} from "../core/files/documentWindowLedger.js";
import {auditDocumentSections} from "../core/files/documentSectionCoverage.js";
import {auditDocumentNumberedReferences}
 from "../core/files/documentNumberedReferenceAudit.js";
import {summarizeSectionReferenceCoverage}
 from "../core/files/documentSectionReferenceCoverage.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
const sha=s=>createHash("sha256").update(s,"utf8").digest("hex");
const source="ВВЕДЕНИЕ\nОбзор литературы [1, с. 4].\n"+
 "ГЛАВА 1. Основы\nВводный аргумент [2, с. 8].\n"+
 "1.1 История\nПервый вопрос [3, с. 6].\n"+
 "1.1.1 Сложная практика\nОсобый случай [2, с. 12].\n"+
 "ЗАКЛЮЧЕНИЕ\nИтог [4].\n"+
 "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\n"+
 "1. Работа первая.\n2. Работа вторая.\n2. Работа другая.\n";
function audit(s){
 const plan=planDocumentSectionWindows(s);
 const ledger=beginWindowLedger(plan,sha(s));
 const sections=auditDocumentSections(plan,ledger,[],[]);
 return {references:auditDocumentNumberedReferences(s,sha(s)),hierarchy:sections.hierarchy};
}
const pages=(s,name)=>{
 const list=[];let offset=0;
 do{const p=pageDocumentText(s,offset);list.push({...p,name});offset=p.nextOffset;}
 while(offset!==null);
 return {name,pages:list};
};
const model=async(_prompt,_task,files)=>{
 const p=createLosslessWindowEvidence(files[0]).fragments[0];
 return JSON.stringify({sections:[{heading:"Проверка",findings:[{
  text:"Автор описывает научную проблему",kind:"author_claim",
  evidence:[{ref:p.ref,quote:p.text.slice(0,Math.min(40,p.text.length))}]
 }]}]});
};

test("counts chapter and descendant references without double-counting direct mentions",()=>{
 const {references,hierarchy}=audit(source);
 const report=summarizeSectionReferenceCoverage(references,hierarchy);
 assert.ok(report);
 assert.equal(report.citationMentions,5);
 assert.equal(report.matched,1);
 assert.equal(report.missing,2);
 assert.equal(report.unresolved,2);
 const chapter=report.sections.find(x=>x.title==="ГЛАВА 1. Основы");
 assert.equal(chapter.citationMentionsDirect,1);
 assert.equal(chapter.citationMentionsSubtree,3);
 assert.equal(chapter.missingSubtree,1);
 assert.equal(chapter.unresolvedSubtree,2);
 const nested=report.sections.find(x=>x.title==="1.1.1 Сложная практика");
 assert.equal(nested.citationMentionsDirect,1);
 assert.equal(nested.citationMentionsSubtree,1);
 const intro=report.sections.find(x=>x.title==="ВВЕДЕНИЕ");
 assert.equal(intro.citationMentionsSubtree,1);
 assert.equal(report.academicSourceQualityVerified,false);
});
test("a missing bibliography remains unresolved, not fabricated missing entries",()=>{
 const text="ВВЕДЕНИЕ\nОснование [8, с. 5].\n";
 const {references,hierarchy}=audit(text);
 const value=summarizeSectionReferenceCoverage(references,hierarchy);
 assert.equal(value.bibliographyAbsent,true);
 assert.equal(value.missing,0);
 assert.equal(value.unresolved,1);
});
test("mutated section identity or mention position invalidates aggregation",()=>{
 const {references,hierarchy}=audit(source);
 assert.equal(summarizeSectionReferenceCoverage(references,{
  ...hierarchy,sections:hierarchy.sections.map((s,i)=>i===2?{...s,parentSectionIndex:99}:s)
 }),null);
 assert.equal(summarizeSectionReferenceCoverage({
  ...references,referenceLinks:{...references.referenceLinks,
   mentions:[{...references.referenceLinks.mentions[0],sourceStart:-1},
    ...references.referenceLinks.mentions.slice(1)]}
 },hierarchy),null);
});
test("end-to-end long document and two-file summaries retain separate chapter counts",async()=>{
 const single=await analyzeDocumentAttachments({task:"Проанализируй",
  attachments:[pages(source,"paper.docx")]},model);
 assert.equal(single.success,true);
 assert.equal(single.referenceSectionCoverage.citationMentions,5);
 const second="ВВЕДЕНИЕ\nДоказательство [9].\n"+
  "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ\n9. Исследование.\n";
 const multi=await analyzeDocumentAttachments({task:"Сопоставь",
  attachments:[pages(source,"same.docx"),pages(second,"same.docx")]},model);
 assert.equal(multi.success,true);
 assert.equal(multi.referenceSectionCoverages.length,2);
 assert.equal(multi.referenceSectionCoverages[0].fileIndex,1);
 assert.equal(multi.referenceSectionCoverages[1].fileIndex,2);
 assert.equal(multi.referenceSectionCoverages[1].matched,1);
});
