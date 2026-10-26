import {getSupabaseClient} from "../../../storage/supabaseClient.js";
import {MODULE_FIELDS} from "./experienceModulePacker.js";
import {hashExperienceModulePart} from "./experienceModuleIntegrity.js";

/* Hydrate a version-pinned facade. Incomplete or mismatched modules
 * are never silently accepted as a successful Experience Skill.
 * Older non-modular Skills continue to load exactly as before.
 */
const TABLE="jessica_experience_skill_modules";
export async function hydrateExperienceModules(skill,skillId,version,client=null) {
  const manifest=skill?.moduleManifest;
  if(!manifest)return skill;
  if(![1,2].includes(manifest.version)||!manifest.counts||
    typeof manifest.counts!=="object"||Array.isArray(manifest.counts))
    throw new Error("Unsupported Skill module manifest");
  const expected=manifest.counts;
  const keys=Object.keys(expected);
  if(!keys.length||keys.some(k=>!MODULE_FIELDS.includes(k)||
    !Number.isInteger(expected[k])||expected[k]<1))throw new Error("Bad Skill module manifest");
  const count=keys.reduce((sum,k)=>sum+expected[k],0);
  if(manifest.version===2){
    const checksums=manifest.sha256;
    if(!checksums||typeof checksums!=="object"||Array.isArray(checksums)||
       Object.keys(checksums).length!==keys.length||
       keys.some(k=>!Array.isArray(checksums[k])||
         checksums[k].length!==expected[k]||
         checksums[k].some(hash=>typeof hash!=="string"||
           !/^[a-f0-9]{64}$/u.test(hash))))
      throw new Error("Skill module checksum manifest invalid");
  }
  if(count>100)throw new Error("Skill module count exceeds limit");
  const {data,error}=await (client||getSupabaseClient()).from(TABLE)
    .select("module_key,part_no,data")
    .eq("skill_id",String(skillId))
    .eq("version",Number(version))
    .order("module_key",{ascending:true})
    .order("part_no",{ascending:true});
  if(error)throw new Error("Skill module load failed: "+error.message);
  if(!Array.isArray(data)||data.length!==count)
    throw new Error("Skill module integrity failure: parts missing or extra");
  const restored={...skill};
  for(const key of keys){
    const rows=data.filter(r=>r.module_key===key);
    if(rows.length!==expected[key])
      throw new Error("Skill module integrity failure: "+key);
    const items=[];
    for(let i=0;i<rows.length;i++){
      if(rows[i].part_no!==i||!Array.isArray(rows[i].data))
        throw new Error("Skill module sequence corrupted: "+key);
      if(manifest.version===2&&
         hashExperienceModulePart(rows[i].data)!==manifest.sha256[key][i])
        throw new Error("Skill module checksum mismatch: "+key+" part "+i);
      items.push(...rows[i].data);
    }
    restored[key]=items;
  }
  delete restored.moduleManifest;
  return restored;
}
