import {verifyDocumentStageReceipt} from "./documentStageReceiptAuth.js";

/**
 * Join a server-confirmed stage series only after source and quote checking.
 * Authentication does not imply scientific, legal, or factual correctness.
 */
export function mergeAuthenticatedDocumentStages(summary,reports,secret) {
    if(!summary||summary.allStageReportsReceived!==true||
       summary.reportOriginAuthenticated!==true||
       summary.modelExecutionAuthenticated!==true||
       summary.reportBodySourceQuotesChecked!==true||
       summary.sourceCoordinatesChecked!==true||
       summary.taskContextMatched!==true||
       summary.reportTextIntegrityChecked!==true||
       summary.sectionCoverage?.allSectionsAuthenticated!==true||
       summary.sectionCoverage?.sectionCount<1||
       !Number.isSafeInteger(summary.stageCount)||
       summary.stageCount<3||summary.stageCount>16||
       !Number.isSafeInteger(summary.totalWindows)||
       summary.windowsReported!==summary.totalWindows||
       !Array.isArray(reports)||reports.length!==summary.stageCount||
       summary.nextStageNumber!==null)return null;
    let bytes=0;
    const texts=[];
    for(let i=0;i<reports.length;i++){
        const entry=reports[i];
        if(entry?.stageReport?.stageNumber!==i+1||
           !verifyDocumentStageReceipt(entry,secret)||
           typeof entry.text!=="string"||!entry.text.trim())
            return null;
        bytes+=Buffer.byteLength(entry.text,"utf8");
        if(bytes>500000)return {success:false,mode:"document",
            stage:"document-stage-aggregate-too-large",
            text:"Отчёты этапов превышают предел единого ответа.",
            readingCoverage:{sourceSha256:summary.sourceSha256,
                processedWindows:summary.totalWindows,
                allStageReportsReceived:true,
                combinedReportAvailable:false,
                completeSourceAnalysisConfirmed:false}};
        texts.push(entry.text);
    }
    return {success:true,mode:"document",
        stage:"document-stages-authenticated",
        text:"Результаты по всем подтверждённым этапам. "+
            "Цитаты сверены с извлечённым исходным текстом. "+
            "Достоверность научных и правовых выводов не подтверждалась.\n\n"+
            texts.join("\n\n────────────────\n\n"),
        readingCoverage:{
            sourceSha256:summary.sourceSha256,
            plannedWindows:summary.totalWindows,
            processedWindows:summary.windowsReported,
            stageCount:summary.stageCount,
            authenticatedStages:summary.stagesReceived,
            allWindowsPresented:true,exactTextCoverageConfirmed:true,
            allStageReportsReceived:true,
            reportOriginAuthenticated:true,
            reportBodySourceQuotesChecked:true,
            claimTruthVerified:false,scientificCorrectnessVerified:false,
            sectionCoverage:summary.sectionCoverage,
            completeSourceAnalysisConfirmed:false}};
}
