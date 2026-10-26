import {createHash} from "node:crypto";
import {planDocumentOverflowBatches} from "./documentOverflowReadBatches.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const LIMIT=16;

/**
 * Validate a prefix of materialized overflow batches against a regenerated
 * complete-source read plan. No unverified prefix may enter AI analysis.
 */
export function verifyMaterializedOverflowBatches(source,sha256,batches){
    if(typeof source!=="string"||!source.length||
       !/^[a-f0-9]{64}$/u.test(String(sha256||""))||
       !Array.isArray(batches)||!batches.length||batches.length>LIMIT)
        return null;
    const plan=planDocumentOverflowBatches(source,sha256);
    if(!plan?.completeSourcePartition||
       batches.length>plan.batchCount||plan.batchCount>LIMIT)
        return null;
    const verifiedWindows=[];
    let cursor=0;
    for(let index=0;index<batches.length;index++){
        const batch=batches[index];
        const canonical=plan.batches[index];
        if(!batch||!canonical||batch.batchNumber!==index+1||
           batch.batchCount!==plan.batchCount||
           batch.totalWindows!==plan.totalWindows||
           batch.sourceSha256!==sha256||
           batch.sourceChars!==source.length||
           batch.sourcePartitionVerified!==true||
           batch.batchMaterialized!==true||
           batch.allBatchesRead!==false||
           batch.modelAnalysisPerformed!==false||
           batch.analyzedWindows!==0||
           batch.resumeImplemented!==false||
           batch.semanticCompletenessConfirmed!==false||
           batch.startOffset!==cursor||
           batch.startOffset!==canonical.startOffset||
           batch.endOffset!==canonical.endOffset||
           batch.windowCount!==canonical.windowCount||
           batch.nextBatchNumber!==(index+1<plan.batchCount?index+2:null)||
           !Array.isArray(batch.windows)||
           batch.windows.length!==canonical.windows.length)return null;
        for(let i=0;i<batch.windows.length;i++){
            const item=batch.windows[i],expected=canonical.windows[i];
            if(!item||item.number!==expected.number||
               item.sectionIndex!==expected.sectionIndex||
               item.title!==expected.title||
               item.startOffset!==cursor||
               item.startOffset!==expected.startOffset||
               item.endOffset!==expected.endOffset||
               item.textSha256!==expected.textSha256||
               item.text!==source.slice(expected.startOffset,expected.endOffset)||
               digest(item.text)!==item.textSha256)return null;
            verifiedWindows.push({...expected,text:item.text});
            cursor=item.endOffset;
        }
        if(cursor!==canonical.endOffset)return null;
    }
    const allCollected=batches.length===plan.batchCount;
    if(allCollected!== (cursor===source.length))return null;
    const base={sourceSha256:sha256,sourceChars:source.length,
        verifiedWindows:verifiedWindows.length,totalWindows:plan.totalWindows,
        verifiedBatches:batches.length,totalBatches:plan.batchCount,
        collectedChars:cursor,nextBatchNumber:allCollected?null:batches.length+1,
        sourcePartitionVerified:true,modelAnalysisPerformed:false,
        analyzedWindows:0,resumeImplemented:false,
        semanticCompletenessConfirmed:false};
    if(!allCollected)return {...base,complete:false,
        allSourceWindowsVerified:false};
    if(verifiedWindows.length!==plan.totalWindows)return null;
    return {...base,complete:true,allSourceWindowsVerified:true,
        windows:verifiedWindows};
}