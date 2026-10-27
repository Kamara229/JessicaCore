// A restarted approval may find the Experience already committed.
// That is successful recovery, not another published Skill version.
export function classifyApprovalPublication(saveResult){
  const existing=saveResult?.existing===true ||
    saveResult?.stage==="already-saved";
  return {
    versionCreated:!existing,
    publicationRecovered:existing
  };
}
