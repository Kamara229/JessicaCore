import {checkStageReportOrder} from "./documentStageReportOrder.js";
import {authenticatedStageResumeStatus} from "./documentStageResumeStatus.js";
import {analyzeVerifiedDocumentStage} from "./documentStageRunner.js";

/**
 * Continue precisely the next stage after a server-attested source-bound prefix.
 * Neither a fabricated client report nor a skipped stage may invoke the model.
 */
export async function continueAuthenticatedDocumentStage(
    {task,source,verified,name,previousReports,stageNumber},generate
){
    const fail=()=>({success:false,mode:"document-stage",
        stage:"document-stage-resume-invalid",
        text:"Для продолжения нужны подлинные результаты всех предыдущих этапов."});
    if(!Number.isSafeInteger(stageNumber)||stageNumber<2||
       !Array.isArray(previousReports)||previousReports.length!==stageNumber-1)
        return fail();
    const coverage=checkStageReportOrder(source,verified,previousReports,task);
    const resume=authenticatedStageResumeStatus(coverage);
    if(!resume||resume.readingCoverage.nextStageNumber!==stageNumber)
        return fail();
    const next=await analyzeVerifiedDocumentStage(
        {task,source,verified,name,stageNumber},generate);
    if(!next.success)return next;
    return {...next,
        continuation:{
            precedingStagesVerified:coverage.stagesReceived,
            precedingWindowsVerified:coverage.windowsReported,
            nextStageNumber:stageNumber===coverage.stageCount?null:stageNumber+1,
            awaitFinalMerge:stageNumber===coverage.stageCount,
            fullSourceAnalysisConfirmed:false,
            scientificCorrectnessVerified:false
        }};
}
