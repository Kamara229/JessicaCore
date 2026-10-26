/**
 * A completed signed prefix can be resumed by explicitly requesting its next
 * stage. Never label partial coverage as a completed document analysis.
 */
export function authenticatedStageResumeStatus(coverage) {
    if(!coverage||coverage.allStageReportsReceived!==false||
       coverage.reportOriginAuthenticated!==true||
       coverage.modelExecutionAuthenticated!==true||
       coverage.sourceCoordinatesChecked!==true||
       coverage.taskContextMatched!==true||
       coverage.reportBodySourceQuotesChecked!==true||
       coverage.reportTextIntegrityChecked!==true||
       !Number.isSafeInteger(coverage.stagesReceived)||
       !Number.isSafeInteger(coverage.stageCount)||
       coverage.stagesReceived<1||coverage.stagesReceived>=coverage.stageCount||
       !Number.isSafeInteger(coverage.nextStageNumber)||
       coverage.nextStageNumber!==coverage.stagesReceived+1||
       !Number.isSafeInteger(coverage.windowsReported)||
       !Number.isSafeInteger(coverage.totalWindows)||
       coverage.windowsReported<1||
       coverage.windowsReported>=coverage.totalWindows||
       !/^[a-f0-9]{64}$/u.test(String(coverage.sourceSha256||"")))
        return null;
    return {success:false,mode:"document",
        stage:"document-stage-resume-ready",
        text:"Проверена завершённая последовательность этапов. Следующий этап: "+
            coverage.nextStageNumber+". Полный анализ ещё не завершён.",
        readingCoverage:{
            ...coverage,
            plannedWindows:coverage.totalWindows,
            processedWindows:coverage.windowsReported,
            authenticatedStages:coverage.stagesReceived,
            nextStageNumber:coverage.nextStageNumber,
            resumeAvailable:true,
            complete:false,
            allWindowsPresented:false,
            completeSourceAnalysisConfirmed:false,
            claimTruthVerified:false
        }};
}
