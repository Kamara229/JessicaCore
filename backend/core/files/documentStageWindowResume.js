import {sealDocumentPagedResume,openDocumentPagedResume}
    from "./documentPagedResumeCheckpoint.js";

/** Reuse the bounded 16-window source/HMAC proof within any 1..16 stage
 * of an independently verified <=256-window long document.
 * Global window numbers and offsets prevent cross-stage checkpoint reuse.
 */
function stagePlan(stage) {
    return stage?.sourceOffsetsVerified===true &&
        stage?.stageIntegrityVerified===true &&
        Number.isSafeInteger(stage.stageNumber) &&
        stage.stageNumber>=1 && stage.stageNumber<=16 &&
        Number.isSafeInteger(stage.stageCount) &&
        stage.stageCount>=stage.stageNumber &&
        stage.stageCount<=16 &&
        Array.isArray(stage.windows) &&
        stage.windows.length===stage.windowCount &&
        stage.windows.length>=2 && stage.windows.length<=16
        ? {complete:true,windows:stage.windows} : null;
}

export function sealStageWindowResume(input) {
    const plan=stagePlan(input?.stage);
    if(!plan)return null;
    return sealDocumentPagedResume({source:input.source,task:input.task,
        name:input.name,plan,rawResponses:input.rawResponses});
}

export function openStageWindowResume(input) {
    const plan=stagePlan(input?.stage);
    if(!plan)return null;
    return openDocumentPagedResume({checkpoint:input.checkpoint,
        source:input.source,task:input.task,name:input.name,plan});
}
