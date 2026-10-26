/**
 * Sanity-check untrusted stage receipt counters without treating them as proof.
 * All counts are bounded to avoid malformed or oversized client metadata.
 */
export function readDocumentStageReceiptStats(receipt){
    const finiteCount=(value,max=10000)=>
        Number.isSafeInteger(value)&&value>=0&&value<=max;
    if(!receipt||typeof receipt!=="object"||
       !finiteCount(receipt.groundedClaims,512)||
       receipt.groundedClaims<1||
       !finiteCount(receipt.rejectedClaims,512)||
       !finiteCount(receipt.anchoredFragments,512)||
       receipt.anchoredFragments<1||
       !finiteCount(receipt.evidenceQuotes,512)||
       receipt.evidenceQuotes<1||
       !finiteCount(receipt.exactLocatedQuotes,512)||
       !finiteCount(receipt.ambiguousQuotes,512)||
       !finiteCount(receipt.nonLiteralQuotes,512)||
       receipt.nonLiteralQuotes!==0||
       receipt.exactLocatedQuotes+receipt.ambiguousQuotes+
           receipt.nonLiteralQuotes!==receipt.evidenceQuotes||
       receipt.claimTruthVerified!==false||
       receipt.independentSourcesVerified!==false)
        return null;
    return {groundedClaims:receipt.groundedClaims,
        rejectedClaims:receipt.rejectedClaims,
        anchoredFragments:receipt.anchoredFragments,
        evidenceQuotes:receipt.evidenceQuotes,
        exactLocatedQuotes:receipt.exactLocatedQuotes,
        ambiguousQuotes:receipt.ambiguousQuotes,
        nonLiteralQuotes:receipt.nonLiteralQuotes};
}
