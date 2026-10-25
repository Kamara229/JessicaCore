/**
 * Execution-grounded Skill attribution. A selected Skill alone is not
 * evidence of use. Require a declared native Skill program and matching
 * successful Tool Results for every experience-marked step.
 */
const asArray=value=>Array.isArray(value)?value:[];
const idOf=skill=>String(skill?.id||skill?.skillId||"").trim();
export function inspectAppliedSkill(context){
  const empty={applied:false,skillIds:[],stepIds:[]};
  if(context?.experience?.used!==true ||
     context?.plan?.experience?.used!==true ||
     context?.runResult?.success!==true)return empty;
  const available=new Set(asArray(context.experience.skills).map(idOf).filter(Boolean));
  const declared=asArray(context.plan.experience.skills).map(idOf)
     .filter(id=>id&&available.has(id));
  if(declared.length===0)return empty;
  const marked=asArray(context.plan.steps)
     .filter(step=>step?.experienceUsed===true && step.id && step.tool);
  if(marked.length===0)return empty;
  const outcomes=asArray(context.runResult.results);
  if(!marked.every(step=>outcomes.some(actual=>
     actual?.id===step.id && actual?.tool===step.tool && actual?.success===true
  )))return empty;
  return {applied:true,skillIds:[...new Set(declared)],stepIds:marked.map(step=>step.id)};
}
