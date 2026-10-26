import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages}
 from "../core/files/documentAnalysisStagePlan.js";
import {checkStageReportOrder} from "../core/files/documentStageReportOrder.js";
import {documentStageTaskSha256} from "../core/files/documentStageTaskIdentity.js";
import {hashDocumentStageReportText} from "../core/files/documentStageReportTextHash.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
const task="Проверь";
const source="ВВЕДЕНИЕ\n"+"Обоснование судебной практики. ".repeat(19500)+
 "\nЗАКЛЮЧЕНИЕ\nВывод.";
const hash=x=>createHash("sha256").update(x,"utf8").digest("hex");
function fixture(){
 const digest=hash(source),first=materializeDocumentOverflowBatch(source,digest,1);
 assert.ok(first);
 const batches=[];
 for(let n=1;n<=first.batchCount;n++)
   batches.push(materializeDocumentOverflowBatch(source,digest,n));
 const verified=verifyMaterializedOverflowBatches(source,digest,batches);
 assert.ok(verified?.complete && verified.totalWindows>32);
 const plan=planVerifiedDocumentAnalysisStages(source,verified);
 const reports=plan.stages.map(stage=>{
  const windows=verified.windows.slice(stage.firstWindowNumber-1,stage.lastWindowNumber);
  const text="Проверенный этап "+stage.stageNumber+" из "+
    plan.stageCount+".\n\n"+windows.map(w=>{
      const quote=source.slice(w.startOffset,w.endOffset)
          .slice(0,50).replace(/\s+/gu," ").trim();
      return "Окно "+w.number+" ("+w.startOffset+"–"+w.endOffset+
          "):\nМатериал\n\n• Позиция автора: Исследуется правовой вопрос "+
          "[О"+w.number+":Ф1:Ф1]\n  Основание: «"+quote+
          "»\nГраницы проверки: точность цитаты не доказывает истинность.";
    }).join("\n\n")+
    "\n\nПроверены цитаты этапа. Полнота всего документа и достоверность выводов не подтверждены.";
  return {success:true,mode:"document-stage",stage:"document-stage-verified",
    text,
    stageReport:{sourceSha256:digest,stageNumber:stage.stageNumber,
     stageCount:plan.stageCount,stageSha256:stage.stageSha256,
     taskSha256:documentStageTaskSha256(task),
     reportTextSha256:hashDocumentStageReportText(text),
     verifiedWindows:windows.length,stageAnalysisComplete:true,
     fullSourceAnalysisConfirmed:false},
    readingCoverage:{sourceSha256:digest,sourceChars:source.length,
     stageNumber:stage.stageNumber,stageCount:plan.stageCount,
     stageSha256:stage.stageSha256,plannedWindows:windows.length,
     taskSha256:documentStageTaskSha256(task),
     processedWindows:windows.length,stageAnalysisComplete:true,
     fullSourceAnalysisConfirmed:false,
     verifiedWindows:windows.map(w=>({windowNumber:w.number,
       sectionIndex:w.sectionIndex,startOffset:w.startOffset,
       endOffset:w.endOffset,windowSha256:w.textSha256,groundedClaims:1,
       rejectedClaims:0,anchoredFragments:1,evidenceQuotes:1,
       exactLocatedQuotes:1,ambiguousQuotes:0,nonLiteralQuotes:0,
       claimTruthVerified:false,independentSourcesVerified:false}))}
  };
 });
 const pages=[];let offset=0;
 do{const p=pageDocumentText(source,offset);pages.push({...p,name:"paper.docx"});offset=p.nextOffset;}
 while(offset!==null);
 return {verified,plan,reports,pages,batches};
}
test("reordered, missing or tampered stage metadata cannot claim analysis",()=>{
 const {verified,reports,plan}=fixture();
 const checked=checkStageReportOrder(source,verified,reports,task);
 assert.equal(checked.allStageReportsReceived,true);
 assert.equal(checked.windowsReported,plan.totalWindows);
 assert.equal(checked.sourceCoordinatesChecked,true);
 assert.equal(checked.taskContextMatched,true);
 assert.equal(checked.reportedGroundedClaims,plan.totalWindows);
 assert.equal(checked.reportedEvidenceQuotes,plan.totalWindows);
 assert.equal(checked.reportedNonExactQuotes,0);
 assert.equal(checked.quoteStatsAuthenticated,false);
 assert.equal(checked.reportTextIntegrityChecked,true);
 assert.equal(checked.reportBodySourceQuotesChecked,true);
 assert.equal(checked.reportOriginAuthenticated,false);
 assert.equal(checkStageReportOrder(source,verified,reports,"Другая задача"),null);
 assert.equal(checked.modelExecutionAuthenticated,false);
 assert.equal(checked.completeSourceAnalysisConfirmed,false);
 assert.equal(checkStageReportOrder(source,verified,reports.slice(0,-1),task).nextStageNumber,
   plan.stageCount);
 assert.equal(checkStageReportOrder(source,verified,[reports[1],reports[0]],task),null);
 const damaged=structuredClone(reports);
 damaged[0].readingCoverage.verifiedWindows[2].startOffset+=1;
 assert.equal(checkStageReportOrder(source,verified,damaged,task),null);
 damaged[0].readingCoverage.verifiedWindows[2].startOffset-=1;
 damaged[0].stageReport.stageSha256="0".repeat(64);
 assert.equal(checkStageReportOrder(source,verified,damaged,task),null);
 const forged=structuredClone(reports);
 forged[0].readingCoverage.taskSha256="0".repeat(64);
 assert.equal(checkStageReportOrder(source,verified,forged,task),null);
});

test("inconsistent quote counters and absent source caveats reject stage reconciliation",()=>{
 const {verified,reports}=fixture();
 const corrupt=[
    row=>{row.evidenceQuotes=2;},
    row=>{row.anchoredFragments=0;},
    row=>{row.exactLocatedQuotes=-1;},
    row=>{row.ambiguousQuotes=1;},
    row=>{row.nonLiteralQuotes=Infinity;},
    row=>{delete row.evidenceQuotes;},
    row=>{row.groundedClaims=0;},
    row=>{row.independentSourcesVerified=true;}
 ];
 for(const change of corrupt){
    const modified=structuredClone(reports);
    change(modified[0].readingCoverage.verifiedWindows[0]);
    assert.equal(checkStageReportOrder(source,verified,modified,task),null);
 }
 const unresolved=structuredClone(reports);
 const receipt=unresolved[0].readingCoverage.verifiedWindows[0];
 receipt.exactLocatedQuotes=0;
 receipt.ambiguousQuotes=1;
 const summary=checkStageReportOrder(source,verified,unresolved,task);
 assert.equal(summary.reportedNonExactQuotes,1);
 assert.equal(summary.quoteStatsAuthenticated,false);
});

test("report integrity rejects altered body, missing digest and oversized stage text",()=>{
 const {verified,reports}=fixture();
 const alterations=[
  r=>{r.text+=" tampered";},
  r=>{r.stageReport.reportTextSha256="0".repeat(64);},
  r=>{delete r.stageReport.reportTextSha256;},
  r=>{delete r.text;},
  r=>{r.text="x".repeat(350001);}
 ];
 for(const change of alterations){
   const altered=structuredClone(reports);
   change(altered[0]);
   assert.equal(checkStageReportOrder(source,verified,altered,task),null);
 }
 const changed=structuredClone(reports);
 changed[0].text+=" modified";
 changed[0].stageReport.reportTextSha256=hashDocumentStageReportText(changed[0].text);
 assert.equal(checkStageReportOrder(source,verified,changed,task),null);
});

test("source quotation, window citation and missing body reject reports despite updated SHA",()=>{
 const {verified,reports}=fixture();
 const transforms=[
   text=>text.replace(/«[^»]+»/u,"«Цитата не из текста»"),
   text=>text.replace("[О1:Ф1:Ф1]","[О2:Ф1:Ф1]"),
   text=>text.replace("Окно 1 (","Окно 99 ("),
   text=>text.replace(/^• /mu,"• Неподтверждено: "),
   text=>text.replace("  Основание: ","  Нет основания: ")
 ];
 for(const change of transforms){
    const modified=structuredClone(reports);
    const first=modified[0];
    const next=change(first.text);
    assert.notEqual(next,first.text);
    first.text=next;
    first.stageReport.reportTextSha256=hashDocumentStageReportText(next);
    assert.equal(checkStageReportOrder(source,verified,modified,task),null);
 }
 const missing=structuredClone(reports);
 const first=missing[0];
 first.text=first.text.replace(/Окно 1 \([^)]*\):\n[\s\S]*?(?=\n\nОкно 2 \()/u,"");
 first.stageReport.reportTextSha256=hashDocumentStageReportText(first.text);
 assert.equal(checkStageReportOrder(source,verified,missing,task),null);
});

test("facade checks report coordinates but never authenticates client model results",async()=>{
 const {reports,pages,batches,plan}=fixture();
 let calls=0;
 const result=await analyzeDocumentAttachments({task:"Проверь",
    attachments:[{name:"paper.docx",pages,overflowBatches:batches,
       stageReports:reports}]},async()=>{calls++;return "{}";});
 assert.equal(calls,0);
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-stage-reports-untrusted");
 assert.equal(result.readingCoverage.stagesReceived,plan.stageCount);
 assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
 assert.equal(result.readingCoverage.modelExecutionAuthenticated,false);
 const bad=await analyzeDocumentAttachments({task:"Проверь",
   attachments:[{name:"paper.docx",pages,overflowBatches:batches,
      stageReports:[reports[1],reports[0]]}]},async()=>{calls++;return "{}";});
 assert.equal(bad.success,false);
 assert.equal(bad.stage,"document-stage-reports-invalid");
 assert.equal(calls,0);
});
