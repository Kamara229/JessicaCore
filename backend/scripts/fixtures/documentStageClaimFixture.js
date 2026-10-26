import {createHash} from "node:crypto";
import {materializeDocumentOverflowBatch as batch} from "../../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches as verify} from "../../core/files/documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages as plan} from "../../core/files/documentAnalysisStagePlan.js";
import {documentStageTaskSha256 as taskSha} from "../../core/files/documentStageTaskIdentity.js";
import {hashDocumentStageReportText as reportSha} from "../../core/files/documentStageReportTextHash.js";
import {signDocumentStageReceipt as sign} from "../../core/files/documentStageReceiptAuth.js";

export const task="Проверь",secret="a".repeat(64);
export const source="ВВЕДЕНИЕ\n"+"Обоснование судебной практики. ".repeat(19500)+
 "\nЗАКЛЮЧЕНИЕ\nВывод.";
const digest=s=>createHash("sha256").update(s,"utf8").digest("hex");
export function makeStageClaimFixture(){
 const hash=digest(source),first=batch(source,hash,1);
 const chunks=Array.from({length:first.batchCount},(_,i)=>batch(source,hash,i+1));
 const verified=verify(source,hash,chunks),steps=plan(source,verified);
 const reports=steps.stages.map(stage=>{
   const windows=verified.windows.slice(stage.firstWindowNumber-1,stage.lastWindowNumber);
   const bodies=windows.map(w=>"Окно "+w.number+" ("+w.startOffset+"–"+w.endOffset+
    "):\nМатериал\n\n• Позиция автора: Анализ норм [О"+w.number+
    ":Ф1:Ф1]\n  Основание: «"+w.text.slice(0,48).replace(/\s+/gu," ").trim()+
    "»\nГраницы проверки: цитата не подтверждает выводы.");
   const text="Проверенный этап "+stage.stageNumber+" из "+steps.stageCount+
    ".\n\n"+bodies.join("\n\n")+
    "\n\nПроверены цитаты этапа. Полнота всего документа и достоверность выводов не подтверждены.";
   const fields={sourceSha256:hash,taskSha256:taskSha(task),
    stageSha256:stage.stageSha256,stageNumber:stage.stageNumber,
    stageCount:steps.stageCount};
   const report={success:true,mode:"document-stage",stage:"document-stage-verified",
    text,stageReport:{...fields,reportTextSha256:reportSha(text),
     verifiedWindows:windows.length,stageAnalysisComplete:true,
     fullSourceAnalysisConfirmed:false},
    readingCoverage:{...fields,sourceChars:source.length,
     plannedWindows:windows.length,processedWindows:windows.length,
     stageAnalysisComplete:true,fullSourceAnalysisConfirmed:false,
     verifiedWindows:windows.map(w=>({windowNumber:w.number,
      sectionIndex:w.sectionIndex,startOffset:w.startOffset,
      endOffset:w.endOffset,windowSha256:w.textSha256,
      groundedClaims:1,rejectedClaims:0,anchoredFragments:1,evidenceQuotes:1,
      exactLocatedQuotes:1,ambiguousQuotes:0,nonLiteralQuotes:0,
      claimTruthVerified:false,independentSourcesVerified:false}))}};
   report.stageReport.authTag=sign(report,secret);
   return report;
 });
 return {verified,reports};
}
