import {getSupabaseClient} from "../../../storage/supabaseClient.js";
import {MODULE_FIELDS} from "./experienceModulePacker.js";

/* Hydrate a version-pinned facade. Incomplete or mismatched modules
 * are never silently accepted as a successful Experience Skill.
 * Older non-modular Skills continue to load exactly as before.
 */
const TABLE="jessica_experience_skill_modules";
export async function hydrateExperienceModules(skill,skillId,version,client=null) {
  const manifest=skill?.moduleManifest;
  if(!manifest)return skill;
  if(manifest.version!==1||!manifest.counts||typeof manifest.counts!=="object")
    throw new Error("Unsupported Skill module manifest");
  const expected=manifest.counts;
  const keys=Object.keys(expected);
  if(!keys.length||keys.some(k=>!MODULE_FIELDS.includes(k)||
    !Number.isInteger(expected[k])||expected[k]<1))throw new Error("Bad Skill module manifest");
  const count=keys.reduce((sum,k)=>sum+expected[k],0);
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
      items.push(...rows[i].data);
    }
    restored[key]=items;
  }
  delete restored.moduleManifest;
  return restored;
}
