import {createHash} from "node:crypto";
import {createLosslessWindowEvidence} from "./documentWindowEvidence.js";
import {validateAndRenderDocumentAnswer} from "./documentAnswerContract.js";
import {locateCitedDocumentFragments} from "./documentCitationSpans.js";
import {locateVerbatimQuoteRanges} from "./documentVerbatimQuotes.js";
import {hasLiteralDocumentStageQuotes} from "./documentStageLiteralQuotes.js";

const sha=text=>createHash("sha256").update(text,"utf8").digest("hex");

/** One quote-grounded window verdict; document source is not persisted. */
export function verifyDocumentStageWindowAnswer(window,name,raw,task) {
    if(!window||typeof window.text!=="string"||!window.text||
       window.text.length>16000||
       !Number.isSafeInteger(window.number)||window.number<1||window.number>256||
       !Number.isSafeInteger(window.startOffset)||window.startOffset<0||
       window.endOffset!==window.startOffset+window.text.length||
       !/^[a-f0-9]{64}$/u.test(String(window.windowSha256||""))||
       sha(window.text)!==window.windowSha256)return null;
    const bundle=createLosslessWindowEvidence({name,text:window.text});
    if(!bundle?.exactSourceCoverage)return null;
    const checked=validateAndRenderDocumentAnswer(raw,bundle,task);
    if(!checked.success||!Number.isSafeInteger(checked.kept)||
       checked.kept<1||!Array.isArray(checked.verifiedFindings)||
       checked.verifiedFindings.length!==checked.kept)return null;
    const spans=locateCitedDocumentFragments(window,bundle,checked);
    const quotes=locateVerbatimQuoteRanges(window,bundle,checked);
    if(!spans?.length||!hasLiteralDocumentStageQuotes(quotes))return null;
    return {
        receipt:{windowNumber:window.number,
            sectionIndex:window.sectionIndex,
            startOffset:window.startOffset,endOffset:window.endOffset,
            windowSha256:window.windowSha256,
            groundedClaims:checked.kept,rejectedClaims:checked.rejected,
            anchoredFragments:spans.length,
            evidenceQuotes:quotes.evidenceQuotes,
            exactLocatedQuotes:quotes.exactLocatedQuotes,
            ambiguousQuotes:quotes.ambiguousQuotes,
            nonLiteralQuotes:quotes.nonLiteralQuotes,
            claimTruthVerified:false,
            independentSourcesVerified:false},
        text:checked.text.replace(/\[Ф1:Ф(\d+)\]/gu,
            (_whole,n)=>"[О"+window.number+":Ф1:Ф"+n+"]")
    };
}
