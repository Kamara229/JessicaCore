// Executing from stale Skills after a failed authoritative read is unsafe.
// Return a terminal, traceable failure before Planner or tool execution.
export function buildExperienceStorageFailure({subtaskId,taskText}) {
  return {
    id:subtaskId,
    text:taskText,
    success:false,
    status:"FAILED",
    stage:"experience-storage",
    result:"Experience Skill не проверен: ошибка загрузки модулей",
    verified:false,
    failure:{
      type:"EXPERIENCE_STORAGE_UNVERIFIED",
      reason:"storage-error"
    },
    executionMeta:{
      experienceIntegrityFailure:true,
      executionStarted:false
    },
    experience:{
      found:false,
      used:false,
      applied:false,
      appliedSkillIds:[],
      source:"experience-storage-error",
      skills:[]
    }
  };
}
