import {createHash} from "node:crypto";
import {buildDocumentSectionIndex} from "./documentSectionIndex.js";
import {selectDocumentSectionPage} from "./documentSectionWindow.js";
import {findVerifiedDocumentPlan,rememberVerifiedDocumentPlan} from "./documentOverflowPlanCache.js";

const digest=value=>createHash("sha256").update(value,"utf8").digest("hex");
const BATCH_SIZE=16;
const MAX_BATCHES=16;

/**
 * Metadata-only blueprint for documents that exceed the 16-window request
 * budget. Every byte of extracted text is assigned to exactly one window.
 * This plans future work ONLY: it does not perform/resume analysis or
 * claim that any model has inspected the windows.
 */
export function planDocumentOverflowBatches(source, expectedSha256) {
    if(typeof source!=="string"||!source.length||
       !/^[a-f0-9]{64}$/u.test(String(expectedSha256||""))||
       digest(source)!==expectedSha256)return null;
    // SHA-256 is recomputed against the actual source on every invocation;
    // cache stores immutable-by-copy structural metadata, not source text.
    const cached=findVerifiedDocumentPlan(expectedSha256,source.length);
    if(cached)return cached;
    const index=buildDocumentSectionIndex(source);
    if(index.complete!==true||!index.sections?.length||
       index.sourceChars!==source.length)return null;
    const batches=[];
    let current=[],cursor=0,totalWindows=0;
    const flush=()=>{
        if(!current.length)return true;
        if(batches.length>=MAX_BATCHES)return false;
        batches.push({
            batchNumber:batches.length+1,
            windowCount:current.length,
            startOffset:current[0].startOffset,
            endOffset:current.at(-1).endOffset,
            firstWindowNumber:current[0].number,
            lastWindowNumber:current.at(-1).number,
            windows:current
        });
        current=[];
        return true;
    };
    for(let i=0;i<index.sections.length;i++){
        const section=index.sections[i];
        if(!Number.isSafeInteger(section.startOffset)||
           !Number.isSafeInteger(section.endOffset)||
           section.startOffset!==cursor||
           section.endOffset<=cursor||section.endOffset>source.length)
            return null;
        let position=cursor;
        while(position<section.endOffset){
            const page=selectDocumentSectionPage(source,index,i,position);
            if(!page||page.offset!==position||!page.text||
               page.text.length>16000)return null;
            const end=position+page.text.length;
            if(end>section.endOffset||end<=position||
               source.slice(position,end)!==page.text)return null;
            if(totalWindows>=BATCH_SIZE*MAX_BATCHES)return null;
            current.push({
                number:++totalWindows,
                sectionIndex:i,
                title:section.title,
                startOffset:position,
                endOffset:end,
                textSha256:digest(page.text)
            });
            position=end;
            if(current.length===BATCH_SIZE && !flush())return null;
        }
        cursor=section.endOffset;
    }
    if(cursor!==source.length||!flush()||!batches.length)return null;
    const plan={sourceSha256:expectedSha256,sourceChars:source.length,
        totalWindows, batchCount:batches.length,
        batches, completeSourcePartition:true,
        analyzedWindows:0,readCompleted:false,
        modelAnalysisPerformed:false,resumeImplemented:false,
        semanticCompletenessConfirmed:false};
    rememberVerifiedDocumentPlan(plan);
    return plan;
}