import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
import {createHash} from "node:crypto";

const digest=value=>createHash("sha256").update(value,"utf8").digest("hex");

/** Verify a complete, ordered set of source windows before calling a model. */
export function beginWindowLedger(plan, sourceSha256) {
    if(plan?.complete!==true||!Array.isArray(plan.windows)||
       !plan.windows.length||plan.windows.length>MAX_VERIFIED_DOCUMENT_WINDOWS||
       !Number.isSafeInteger(plan.sourceChars)||plan.sourceChars<=0||
       !/^[a-f0-9]{64}$/u.test(String(sourceSha256||"")))return null;
    let offset=0;
    const sourceHash=createHash("sha256");
    const expected=[];
    for(const window of plan.windows){
        if(window?.number!==expected.length+1||
           !Number.isSafeInteger(window.startOffset)||
           window.startOffset!==offset||
           !Number.isSafeInteger(window.endOffset)||
           typeof window.text!=="string"||!window.text.length||
           window.text.length>16000||
           window.endOffset!==offset+window.text.length||
           !Number.isSafeInteger(window.sectionIndex)||
           window.sectionIndex<0)return null;
        sourceHash.update(window.text,"utf8");
        expected.push({number:window.number,sectionIndex:window.sectionIndex,
            startOffset:window.startOffset,endOffset:window.endOffset,
            windowSha256:digest(window.text)});
        offset=window.endOffset;
    }
    if(offset!==plan.sourceChars||sourceHash.digest("hex")!==sourceSha256)return null;
    return {sourceSha256,sourceChars:offset,expected,confirmed:[]};
}

/** Record only a window whose model response passed quote verification. */
export function noteVerifiedWindow(ledger,window,checked){
    const target=ledger?.expected?.[ledger.confirmed.length];
    if(!target||!window||checked?.success!==true||
       !Number.isSafeInteger(checked.kept)||checked.kept<1||
       !Number.isSafeInteger(checked.rejected)||checked.rejected<0||
       window.number!==target.number||
       window.sectionIndex!==target.sectionIndex||
       window.startOffset!==target.startOffset||
       window.endOffset!==target.endOffset||
       digest(window.text)!==target.windowSha256)return false;
    ledger.confirmed.push({...target,groundedClaims:checked.kept,
        rejectedClaims:checked.rejected,
        citedFragments:new Set(checked.citedRefs||[]).size});
    return true;
}

/** Output contains only offsets, hashes, and counts, not user document text. */
export function windowLedgerSummary(ledger) {
    if(!ledger)return null;
    return {sourceSha256:ledger.sourceSha256,sourceChars:ledger.sourceChars,
        plannedWindows:ledger.expected.length,
        processedWindows:ledger.confirmed.length,
        verifiedWindows:ledger.confirmed.map(w=>({...w})),
        allWindowsPresented:ledger.confirmed.length===ledger.expected.length,
        exactTextCoverageConfirmed:true,
        completeSourceAnalysisConfirmed:false};
}
