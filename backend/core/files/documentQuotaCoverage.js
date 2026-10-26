/**
 * Explicitly preserve verified window progress when the provider reaches 429.
 * Metadata is diagnostic only. There is no durable checkpoint or auto-resume.
 */
export function markDocumentQuotaInterruption(error, ledgerSummary, interruptedWindow) {
    if (error?.status!==429 || !ledgerSummary ||
        !Number.isSafeInteger(interruptedWindow) || interruptedWindow<1)
        return error;
    const coverage={...ledgerSummary,complete:false,
        allWindowsPresented:false,interruptedWindow,
        interruption:"ai-provider-quota",
        resumeAvailable:ledgerSummary.resumeAvailable===true &&
            !!ledgerSummary.resumeCheckpoint,
        completeSourceAnalysisConfirmed:false};
    error.documentReadingCoverage=coverage;
    return error;
}

/** Fold confirmed per-file progress into the bounded multi-document plan. */
export function markMultipleDocumentQuotaInterruption(
    error,plan,file,finishedFiles,finishedWindows
){
    if(error?.status!==429 || !plan || !file ||
        !Array.isArray(finishedFiles) || !Number.isSafeInteger(finishedWindows) ||
        finishedWindows<0)return error;
    const partial=error.documentReadingCoverage;
    const local=Number.isSafeInteger(partial?.processedWindows)
        ? partial.processedWindows:0;
    if(local>file.plannedWindows || finishedWindows+local>plan.totalWindows)return error;
    error.documentReadingCoverage={
        plannedWindows:plan.totalWindows,
        processedWindows:finishedWindows+local,
        files:finishedFiles.map(x=>({...x})),
        incompleteFile:{fileIndex:file.index,name:file.name,
            sourceSha256:file.sourceSha256,
            plannedWindows:file.plannedWindows,
            processedWindows:local,
            ...(partial||{})},
        complete:false,allWindowsPresented:false,
        completeSourceAnalysisConfirmed:false,
        interruption:"ai-provider-quota",resumeAvailable:false
    };
    return error;
}

/** Only metadata is returned; never disguise 429 as a verified answer. */
export function addInterruptedDocumentCoverage(result,error) {
    const coverage=error?.documentReadingCoverage;
    return coverage ? {...result,readingCoverage:coverage}:result;
}
