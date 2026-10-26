const TABLES={
  skills:"jessica_experience_skills",
  history:"jessica_experience_skill_versions",
  modules:"jessica_experience_skill_modules",
  audit:"jessica_experience_skill_audit",
  evidence:"jessica_experience_evidence"
};
const prefix="jessica-rpc-race-probe-";
export async function countExperienceRaceRows(client,sid){
  if(typeof sid!=="string"||!sid.startsWith(prefix))
    throw Error("Refusing to inspect non-disposable Skill");
  const entries=await Promise.all(Object.entries(TABLES).map(async([key,table])=>{
    const field=key==="skills"?"id":"skill_id";
    const {count,error}=await client.from(table).select("*",{head:true,count:"exact"}).eq(field,sid);
    if(error||!Number.isInteger(count))throw Error("Race row count "+table+": "+(error?.message||"unknown"));
    return [key,count];
  }));
  return Object.fromEntries(entries);
}
export async function deleteExperienceRaceRows(client,sid){
  if(typeof sid!=="string"||!/^jessica-rpc-race-probe-[\da-f-]{36}$/u.test(sid))
    throw Error("Refusing to delete a non-disposable Skill");
  // Current row removal cascades immutable history and module parts;
  // audit and evidence have their own lifecycle and must be scoped.
  for(const table of [TABLES.audit,TABLES.evidence]){
    const {error}=await client.from(table).delete().eq("skill_id",sid);
    if(error)throw Error("Race cleanup "+table+": "+error.message);
  }
  const {error}=await client.from(TABLES.skills).delete().eq("id",sid);
  if(error)throw Error("Race cleanup Skill: "+error.message);
  const remaining=await countExperienceRaceRows(client,sid);
  if(Object.values(remaining).some(count=>count!==0))
    throw Error("Race cleanup left disposable rows: "+JSON.stringify(remaining));
  return remaining;
}
