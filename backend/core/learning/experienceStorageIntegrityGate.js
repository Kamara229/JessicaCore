// Cross-layer fail-closed policy: no reinforcement after an unverifiable
// Supabase Skill read. Check both single and complex outer execution traces.
function isStorageFailure(result){
  return Boolean(result?.failure?.type==="EXPERIENCE_STORAGE_UNVERIFIED" ||
    result?.executionMeta?.experienceIntegrityFailure===true ||
    result?.stage==="experience-storage");
}
export function hasExperienceStorageIntegrityFailure(trace){
  if(!trace || typeof trace!=="object")return false;
  if(isStorageFailure(trace.result))return true;
  if(Array.isArray(trace?.result?.results) &&
     trace.result.results.some(isStorageFailure))return true;
  return Array.isArray(trace.failures) &&
    trace.failures.some(f=>f?.type==="EXPERIENCE_STORAGE_UNVERIFIED");
}
