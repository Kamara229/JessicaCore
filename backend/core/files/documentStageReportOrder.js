import {planVerifiedDocumentAnalysisStages} from "./documentAnalysisStagePlan.js";
import {matchesDocumentStageTask} from "./documentStageTaskIdentity.js";
import {readDocumentStageReceiptStats} from "./documentStageReceiptStats.js";
import {matchesDocumentStageReportText} from "./documentStageReportTextHash.js";
import {inspectDocumentStageReportBody} from "./documentStageReportBodyAudit.js";

/** Metadata consistency only. Client stage reports are NOT authenticated. */
export function checkStageReportOrder(source,verified,reports,task){
 const plan=planVerifiedDocumentAnalysisStages(source,verified);
 if(!plan||!Array.isArray(reports)||!reports.length||
    reports.length>plan.stageCount)return null;
 let count=0,offset=0,claims=0,quotes=0,unlocated=0,anchors=0;
 for(let i=0;i<reports.length;i++){
  const r=reports[i],stage=plan.stages[i],s=r?.stageReport,
        c=r?.readingCoverage;
  if(!matchesDocumentStageReportText(r)||
     !matchesDocumentStageTask(task,s,c)||
     r?.success!==true||r?.mode!=="document-stage"||
     r?.stage!=="document-stage-verified"||
     s?.stageNumber!==i+1||s?.stageCount!==plan.stageCount||
     s?.stageSha256!==stage.stageSha256||
     s?.sourceSha256!==plan.sourceSha256||
     s?.verifiedWindows!==stage.windowCount||
     s?.stageAnalysisComplete!==true||
     s?.fullSourceAnalysisConfirmed!==false||
     c?.stageNumber!==i+1||c?.stageCount!==plan.stageCount||
     c?.stageSha256!==stage.stageSha256||
     c?.sourceSha256!==plan.sourceSha256||
     c?.sourceChars!==source.length||
     c?.processedWindows!==stage.windowCount||
     c?.plannedWindows!==stage.windowCount||
     c?.stageAnalysisComplete!==true||
     c?.fullSourceAnalysisConfirmed!==false||
     !Array.isArray(c?.verifiedWindows)||
     c.verifiedWindows.length!==stage.windowCount)return null;
  const prevClaims=claims,prevQuotes=quotes,prevAnchors=anchors;
  for(const record of c.verifiedWindows){
   const w=verified.windows[count];
   if(!w||!record||record.windowNumber!==w.number||
      record.sectionIndex!==w.sectionIndex||
      record.startOffset!==offset||
      record.endOffset!==w.endOffset||
      record.windowSha256!==w.textSha256||
      !Number.isSafeInteger(record.groundedClaims)||
      record.groundedClaims<1||record.groundedClaims>14||
      record.claimTruthVerified!==false||
      record.independentSourcesVerified!==false)return null;
   const stats=readDocumentStageReceiptStats(record);
   if(!stats)return null;
   claims+=stats.groundedClaims;
   quotes+=stats.evidenceQuotes;
   anchors+=stats.anchoredFragments;
   unlocated+=stats.ambiguousQuotes+stats.nonLiteralQuotes;
   offset=w.endOffset;
   count++;
  }
  if(offset!==stage.endOffset)return null;
  const sourceAudit=inspectDocumentStageReportBody(r,{
     ...stage,stageCount:plan.stageCount,
     windows:verified.windows.slice(
        stage.firstWindowNumber-1,stage.lastWindowNumber)
  },source);
  if(!sourceAudit||sourceAudit.bodyClaims!==claims-prevClaims||
     sourceAudit.bodyQuotes!==quotes-prevQuotes||
     sourceAudit.bodyUniqueRefs!==anchors-prevAnchors)return null;
 }
 const all=reports.length===plan.stageCount;
 if(all!==(offset===source.length))return null;
 return {sourceSha256:plan.sourceSha256,stageCount:plan.stageCount,
  stagesReceived:reports.length,windowsReported:count,
  totalWindows:plan.totalWindows,
  nextStageNumber:all?null:reports.length+1,
  allStageReportsReceived:all,sourceCoordinatesChecked:true,
  taskContextMatched:true,
  reportedGroundedClaims:claims,reportedEvidenceQuotes:quotes,
  reportedAnchoredFragments:anchors,
  reportedNonExactQuotes:unlocated,quoteStatsAuthenticated:false,
  reportTextIntegrityChecked:true,reportOriginAuthenticated:false,
  reportBodySourceQuotesChecked:true,
  modelExecutionAuthenticated:false,claimTruthVerified:false,
  completeSourceAnalysisConfirmed:false};
}
