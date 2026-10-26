import {createHash} from "node:crypto";
import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";

const hash=s=>createHash("sha256").update(s,"utf8").digest("hex");
const MAX_SOURCE_WINDOWS=256;
const STAGE_WINDOWS=16;

/** Source-only stage manifest for >32 windows; no AI work is asserted. */
export function planVerifiedDocumentAnalysisStages(source,verified) {
    if(typeof source!=="string"||!source.length||source.length>4_000_000||
       !verified||verified.complete!==true||
       verified.allSourceWindowsVerified!==true||
       verified.modelAnalysisPerformed!==false||
       verified.analyzedWindows!==0||
       verified.semanticCompletenessConfirmed!==false||
       !/^[a-f0-9]{64}$/u.test(String(verified.sourceSha256||""))||
       hash(source)!==verified.sourceSha256||
       verified.sourceChars!==source.length||
       !Array.isArray(verified.windows)||
       verified.windows.length<=MAX_VERIFIED_DOCUMENT_WINDOWS||
       verified.windows.length>MAX_SOURCE_WINDOWS||
       verified.totalWindows!==verified.windows.length||
       verified.verifiedWindows!==verified.windows.length)return null;
    let at=0;
    const stages=[];
    let batch=[];
    for(const w of verified.windows) {
        if(w?.number!==at+1||
           !Number.isSafeInteger(w.startOffset)||
           !Number.isSafeInteger(w.endOffset)||
           typeof w.text!=="string"||!w.text.length||
           w.text.length>16000||
           w.endOffset!==w.startOffset+w.text.length||
           (batch.length && w.startOffset!==batch.at(-1).endOffset)||
           (!batch.length && w.startOffset!==(stages.at(-1)?.endOffset??0))||
           !/^[a-f0-9]{64}$/u.test(String(w.textSha256||""))||
           hash(w.text)!==w.textSha256||
           source.slice(w.startOffset,w.endOffset)!==w.text||
           !Number.isSafeInteger(w.sectionIndex)||w.sectionIndex<0||
           typeof w.title!=="string"||!w.title.trim())
            return null;
        at++;
        batch.push(w);
        if(batch.length===STAGE_WINDOWS||
           at===verified.windows.length){
            const first=batch[0],last=batch.at(-1);
            stages.push({stageNumber:stages.length+1,
                firstWindowNumber:first.number,lastWindowNumber:last.number,
                windowCount:batch.length,
                startOffset:first.startOffset,endOffset:last.endOffset,
                stageSha256:hash(source.slice(first.startOffset,last.endOffset)),
                sectionIndexes:[...new Set(batch.map(x=>x.sectionIndex))]});
            batch=[];
        }
    }
    if(stages.at(-1)?.endOffset!==source.length||
       stages.reduce((n,s)=>n+s.windowCount,0)!==verified.totalWindows)return null;
    return {sourceSha256:verified.sourceSha256,
        sourceChars:source.length,totalWindows:verified.totalWindows,
        stageCount:stages.length,stages,
        nextStageNumber:1,verifiedSourceWindows:verified.windows.length,
        processedWindows:0,completedStages:0,
        stageReadPlanVerified:true,modelAnalysisPerformed:false,
        resumeImplemented:false,semanticCompletenessConfirmed:false};
}