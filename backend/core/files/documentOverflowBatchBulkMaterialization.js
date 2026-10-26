import {prepareDocumentOverflowMaterialization}
 from "./documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "./documentOverflowBatchVerification.js";

/** Bounded one-request bulk path; independently verifies all emitted windows. */
export function materializeVerifiedDocumentOverflowBatches(source,sourceSha256){
    const session=prepareDocumentOverflowMaterialization(source,sourceSha256);
    if(!session)return null;
    const batches=[];
    for(let number=1;number<=session.batchCount;number++){
        const batch=session.materialize(number);
        if(!batch)return null;
        batches.push(batch);
    }
    const verified=verifyMaterializedOverflowBatches(
        source,sourceSha256,batches);
    if(!verified?.complete||verified.totalWindows!==session.totalWindows)
        return null;
    return {batches,verified,sourceSha256,
        sourcePartitionVerified:true,
        independentBatchVerificationPerformed:true,
        modelAnalysisPerformed:false,semanticCompletenessConfirmed:false};
}
