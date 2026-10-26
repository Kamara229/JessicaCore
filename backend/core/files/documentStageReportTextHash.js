import {createHash} from "node:crypto";

const MAX_STAGE_REPORT_BYTES=350000;

/**
 * Integrity checksum for a stage's rendered answer (not authentication).
 * Fail closed on missing, nontextual or oversized content.
 */
export function hashDocumentStageReportText(value){
    if(typeof value!=="string"||!value.trim()||
       Buffer.byteLength(value,"utf8")>MAX_STAGE_REPORT_BYTES)return null;
    return createHash("sha256").update(value,"utf8").digest("hex");
}

export function matchesDocumentStageReportText(report){
    const actual=hashDocumentStageReportText(report?.text);
    return actual!==null&&report?.stageReport?.reportTextSha256===actual;
}
