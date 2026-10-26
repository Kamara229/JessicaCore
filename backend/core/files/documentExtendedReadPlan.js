import {createHash} from "node:crypto";
import {buildDocumentSectionIndex} from "./documentSectionIndex.js";
import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";

const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");

/**
 * Convert a verified complete batch series into an analysis-ready plan.
 * A source digest and every source-derived title/offset must still match.
 */
export function buildVerifiedExtendedReadPlan(source,expectedHash,verified){
    if(typeof source!=="string"||!source.length||
       !/^[a-f0-9]{64}$/u.test(String(expectedHash||""))||
       digest(source)!==expectedHash||
       verified?.complete!==true||
       verified.allSourceWindowsVerified!==true||
       verified.sourceSha256!==expectedHash||
       verified.sourceChars!==source.length||
       !Array.isArray(verified.windows)||
       verified.windows.length<=16||
       verified.windows.length>MAX_VERIFIED_DOCUMENT_WINDOWS||
       verified.totalWindows!==verified.windows.length)
        return null;
    const index=buildDocumentSectionIndex(source);
    if(!index.complete||index.sourceChars!==source.length)
        return null;
    let offset=0,lastSection=-1;
    const windows=[];
    for(const item of verified.windows){
        const section=index.sections[item?.sectionIndex];
        if(!section||!Number.isSafeInteger(item.number)||
           item.number!==windows.length+1||
           item.sectionIndex<lastSection||
           item.title!==section.title||
           !Number.isSafeInteger(item.startOffset)||
           item.startOffset!==offset||
           item.startOffset<section.startOffset||
           !Number.isSafeInteger(item.endOffset)||
           item.endOffset>section.endOffset||
           item.endOffset<=offset||
           typeof item.text!=="string"||
           item.text.length>16000||
           item.text!==source.slice(offset,item.endOffset)||
           digest(item.text)!==item.textSha256)return null;
        offset=item.endOffset;
        lastSection=item.sectionIndex;
        windows.push({number:item.number,sectionIndex:item.sectionIndex,
            title:item.title,startOffset:item.startOffset,endOffset:item.endOffset,
            text:item.text});
    }
    if(offset!==source.length)return null;
    return {sourceChars:source.length,windows,complete:true,
        reason:null,expanded:true,batchCount:verified.verifiedBatches};
}
