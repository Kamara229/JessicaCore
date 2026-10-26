import {createHmac,timingSafeEqual} from "node:crypto";
import {matchesDocumentStageReportText} from "./documentStageReportTextHash.js";

const HEX=/^[a-f0-9]{64}$/u;

/** Stateless, server-only provenance for a fully source-checked stage receipt. */
function serializeStageReceipt(report){
 const s=report?.stageReport,c=report?.readingCoverage;
 if(report?.success!==true||report.mode!=="document-stage"||
    report.stage!=="document-stage-verified"||
    !s||!c||!matchesDocumentStageReportText(report)||
    s.stageAnalysisComplete!==true||c.stageAnalysisComplete!==true||
    s.fullSourceAnalysisConfirmed!==false||
    c.fullSourceAnalysisConfirmed!==false||
    !Number.isSafeInteger(s.stageNumber)||s.stageNumber<1||
    s.stageNumber>16||!Number.isSafeInteger(s.verifiedWindows)||
    s.verifiedWindows<1||s.verifiedWindows>16||
    !Array.isArray(c.verifiedWindows)||
    c.verifiedWindows.length!==s.verifiedWindows)return null;
 const hashes=[s.sourceSha256,s.taskSha256,s.stageSha256,
   s.reportTextSha256,c.sourceSha256,c.taskSha256,c.stageSha256];
 if(hashes.some(x=>typeof x!=="string"||!HEX.test(x)))return null;
 const windows=c.verifiedWindows.map(w=>[
   w.windowNumber,w.sectionIndex,w.startOffset,w.endOffset,
   w.windowSha256,w.groundedClaims,w.rejectedClaims,w.anchoredFragments,
   w.evidenceQuotes,w.exactLocatedQuotes,w.ambiguousQuotes,
   w.nonLiteralQuotes,w.claimTruthVerified,w.independentSourcesVerified]);
 const payload=JSON.stringify(["Jessica.Stage.v1",
   s.sourceSha256,s.taskSha256,s.stageSha256,s.reportTextSha256,
   s.stageNumber,s.stageCount,s.verifiedWindows,
   c.sourceSha256,c.taskSha256,c.stageSha256,c.sourceChars,
   c.stageNumber,c.stageCount,c.totalSourceWindows,
   c.plannedWindows,c.processedWindows,
   c.stageAnalysisComplete,c.fullSourceAnalysisConfirmed,windows]);
 return Buffer.byteLength(payload,"utf8")<=64000?payload:null;
}
export function signDocumentStageReceipt(report,secret=process.env.JESSICA_STAGE_ATTESTATION_KEY){
 if(typeof secret!=="string"||!HEX.test(secret))return null;
 const payload=serializeStageReceipt(report);
 if(!payload)return null;
 return "v1:"+createHmac("sha256",Buffer.from(secret,"hex"))
   .update(payload,"utf8").digest("hex");
}
export function verifyDocumentStageReceipt(report,secret=process.env.JESSICA_STAGE_ATTESTATION_KEY){
 const tag=report?.stageReport?.authTag;
 if(typeof tag!=="string"||!/^v1:[a-f0-9]{64}$/u.test(tag))return false;
 const expected=signDocumentStageReceipt(report,secret);
 return typeof expected==="string"&&
    timingSafeEqual(Buffer.from(tag,"utf8"),Buffer.from(expected,"utf8"));
}
