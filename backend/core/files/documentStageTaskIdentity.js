import {createHash} from "node:crypto";

/** Bind a stage receipt to task text, not to a user or a provider identity. */
export function documentStageTaskSha256(task){
    if(typeof task!=="string")return null;
    const normalized=task.trim().slice(0,3000);
    if(!normalized)return null;
    return createHash("sha256").update(normalized,"utf8").digest("hex");
}

export function matchesDocumentStageTask(task,stageReport,readingCoverage){
    const expected=documentStageTaskSha256(task);
    return expected!==null&&stageReport?.taskSha256===expected&&
        readingCoverage?.taskSha256===expected;
}
