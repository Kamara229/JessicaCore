import {createHash} from "node:crypto";
import {planVerifiedDocumentAnalysisStages} from "./documentAnalysisStagePlan.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const MAX_STAGE_UTF8_BYTES=650000;

/**
 * Expose only one source-verified, contiguous processing stage.
 * Does not run a model, store source text or mark any stage analyzed.
 */
export function materializeVerifiedDocumentAnalysisStage(source,verified,stageNumber){
    if(!Number.isSafeInteger(stageNumber)||stageNumber<1)return null;
    const plan=planVerifiedDocumentAnalysisStages(source,verified);
    const stage=plan?.stages?.[stageNumber-1];
    if(!stage||stage.stageNumber!==stageNumber||
       stage.windowCount<1||stage.windowCount>16||
       stage.firstWindowNumber<1||
       stage.lastWindowNumber!==stage.firstWindowNumber+stage.windowCount-1)
        return null;
    const windows=verified.windows.slice(
        stage.firstWindowNumber-1,stage.lastWindowNumber);
    if(windows.length!==stage.windowCount)return null;
    let cursor=stage.startOffset,bytes=0;
    const payload=[];
    for(let i=0;i<windows.length;i++){
        const entry=windows[i];
        if(!entry||entry.number!==stage.firstWindowNumber+i||
           entry.startOffset!==cursor||
           entry.endOffset<=cursor||
           entry.text!==source.slice(entry.startOffset,entry.endOffset)||
           digest(entry.text)!==entry.textSha256)return null;
        bytes+=Buffer.byteLength(entry.text,"utf8");
        if(bytes>MAX_STAGE_UTF8_BYTES)return null;
        payload.push({number:entry.number,sectionIndex:entry.sectionIndex,
            title:entry.title,startOffset:entry.startOffset,
            endOffset:entry.endOffset,
            windowSha256:entry.textSha256,text:entry.text});
        cursor=entry.endOffset;
    }
    const segment=source.slice(stage.startOffset,stage.endOffset);
    if(cursor!==stage.endOffset||
       digest(segment)!==stage.stageSha256||
       payload.map(w=>w.text).join("")!==segment)return null;
    return {sourceSha256:plan.sourceSha256,
        sourceChars:plan.sourceChars,
        stageNumber,stageCount:plan.stageCount,
        stageSha256:stage.stageSha256,
        firstWindowNumber:stage.firstWindowNumber,
        lastWindowNumber:stage.lastWindowNumber,
        windowCount:stage.windowCount,
        startOffset:stage.startOffset,endOffset:stage.endOffset,
        nextStageNumber:stageNumber===plan.stageCount?null:stageNumber+1,
        textBytes:bytes,
        windows:payload,
        sourceOffsetsVerified:true,
        stageIntegrityVerified:true,
        modelAnalysisPerformed:false,stageAnalysisComplete:false,
        fullSourceAnalysisConfirmed:false};
}