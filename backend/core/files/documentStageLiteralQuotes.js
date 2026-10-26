/** Exact source text is required for stage citations; ambiguity is disclosed. */
export function hasLiteralDocumentStageQuotes(quoteAudit){
    if(!quoteAudit||
       !Number.isSafeInteger(quoteAudit.evidenceQuotes)||
       quoteAudit.evidenceQuotes<1||
       !Number.isSafeInteger(quoteAudit.exactLocatedQuotes)||
       quoteAudit.exactLocatedQuotes<0||
       !Number.isSafeInteger(quoteAudit.ambiguousQuotes)||
       quoteAudit.ambiguousQuotes<0||
       quoteAudit.nonLiteralQuotes!==0)return false;
    return quoteAudit.exactLocatedQuotes+
        quoteAudit.ambiguousQuotes===quoteAudit.evidenceQuotes;
}
