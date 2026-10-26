// Fail closed if the authoritative Skill store cannot verify its modules.
// Stale Decomposer memoryHints are never a replacement for corrupt storage.
export function mergeResolvedSubtaskExperience(resolved,memoryExperience){
  if(resolved?.reason==="storage-error"||resolved?.source==="error"){
    return {
      found:false,
      confidence:0,
      source:"experience-storage-error",
      reason:"storage-error",
      experience:null,
      planningContext:{
        experience:null,
        sourceRules:[],
        constraints:[],
        instructions:[],
        plannerHints:[],
        metadata:{experienceFound:false,reason:"storage-error"}
      }
    };
  }
  if(resolved?.found===true&&resolved.experience)return resolved;
  if(memoryExperience){
    return {
      found:true,confidence:0.8,source:"memory-context",
      experience:memoryExperience,
      planningContext:resolved?.planningContext||null
    };
  }
  return resolved||{
    found:false,confidence:0,source:"none",
    experience:null,planningContext:null
  };
}
