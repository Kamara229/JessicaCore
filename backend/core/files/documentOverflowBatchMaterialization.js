import {createHash} from "node:crypto";
import {planDocumentOverflowBatches} from "./documentOverflowReadBatches.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");

/** Only called with a locally recomputed SHA-256 plan, never caller metadata. */
function materializeFromVerifiedPlan(source,sourceSha256,batchNumber,plan){
    if(!Number.isSafeInteger(batchNumber)||batchNumber<1||
       batchNumber>plan.batchCount)return null;
    const batch=plan.batches[batchNumber-1];
    if(!batch||batch.batchNumber!==batchNumber||
       !Array.isArray(batch.windows)||!batch.windows.length||
       batch.windows.length>16||batch.windowCount!==batch.windows.length)
        return null;
    let cursor=batch.startOffset;
    const windows=[];
    for(const window of batch.windows){
        if(window.startOffset!==cursor||
           !Number.isSafeInteger(window.endOffset)||
           window.endOffset<=cursor||window.endOffset>batch.endOffset||
           window.endOffset-window.startOffset>16000)return null;
        const text=source.slice(window.startOffset,window.endOffset);
        if(!text||digest(text)!==window.textSha256||
           /[\uD800-\uDBFF]$/u.test(text)||
           /^[\uDC00-\uDFFF]/u.test(text))return null;
        windows.push({...window,text});
        cursor=window.endOffset;
    }
    if(cursor!==batch.endOffset)return null;
    return {sourceSha256,sourceChars:source.length,
        batchNumber,batchCount:plan.batchCount,
        totalWindows:plan.totalWindows,
        windowCount:windows.length,
        startOffset:batch.startOffset,endOffset:batch.endOffset,
        nextBatchNumber:batchNumber<plan.batchCount?batchNumber+1:null,
        windows,sourcePartitionVerified:true,
        batchMaterialized:true,
        allBatchesRead:false,modelAnalysisPerformed:false,
        analyzedWindows:0,resumeImplemented:false,
        semanticCompletenessConfirmed:false};
}

/** Independent request: always validate source digest and partition. */
export function materializeDocumentOverflowBatch(source,sourceSha256,batchNumber){
    if(typeof source!=="string"||!source.length||
       !/^[a-f0-9]{64}$/u.test(String(sourceSha256||""))||
       !Number.isSafeInteger(batchNumber)||batchNumber<1)return null;
    const plan=planDocumentOverflowBatches(source,sourceSha256);
    if(!plan?.completeSourcePartition)return null;
    return materializeFromVerifiedPlan(
        source,sourceSha256,batchNumber,plan);
}

/** One-request bounded session: source is immutable, plan stays private.
 * Never accept a client-provided plan. Independently verify emitted batches.
 */
export function prepareDocumentOverflowMaterialization(source,sourceSha256){
    if(typeof source!=="string"||!source.length||
       !/^[a-f0-9]{64}$/u.test(String(sourceSha256||"")))return null;
    const plan=planDocumentOverflowBatches(source,sourceSha256);
    if(!plan?.completeSourcePartition||plan.batchCount>16||
       plan.totalWindows>256)return null;
    return Object.freeze({
        sourceSha256,sourceChars:source.length,
        batchCount:plan.batchCount,totalWindows:plan.totalWindows,
        materialize(batchNumber){
            return materializeFromVerifiedPlan(
                source,sourceSha256,batchNumber,plan);
        }
    });
}
