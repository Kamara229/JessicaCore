import {getSupabaseClient} from "../../../storage/supabaseClient.js";
import {packageExperienceModules} from "./experienceModulePacker.js";

/*
 * Save one Skill facade and all functional modules in ONE Postgres RPC.
 * No partial skill is published if a module fails to persist. Preserve
 * v3 for legacy Skills without modular fields.
 */
export async function saveModuleAwareExperience({
  skillId,version,previousVersion,payload,metadata,evidence
}) {
  const {facade,modules}=packageExperienceModules(payload);
  const args={
    p_skill_id:skillId,
    p_version:version,
    p_previous_version:previousVersion,
    p_mode:metadata.mode||"create",
    p_enabled:facade.enabled,
    p_payload:facade,
    p_metadata:metadata,
    p_evidence:evidence
  };
  const modular=modules.length>0;
  if(modular)args.p_modules=modules;
  const rpc=modular?"save_jessica_experience_skill_v4":
    "save_jessica_experience_skill_v3";
  const {data,error}=await getSupabaseClient().rpc(rpc,args);
  return {data,error,modular,facade,modulesSaved:modules.length};
}
