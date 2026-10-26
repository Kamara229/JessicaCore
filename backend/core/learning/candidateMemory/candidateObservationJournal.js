import {getSupabaseClient} from "../../../storage/supabaseClient.js";

// Transactional write: Candidate row and per-queue journal commit or roll back together.
// On replay return persisted state, never apply the merge a second time.
export async function applyCandidateObservation({
  queueItemId,id,candidateKey,action,skillId=null,category="general",
  candidate,metadata={},expectedOccurrences=null,client=null
}={}){
  if(!queueItemId||!id||!candidateKey||
     !["NEW","EXACT","SIMILARITY"].includes(action)||
     !candidate||typeof candidate!=="object"||Array.isArray(candidate)){
    throw new Error("Candidate observation requires queue ID and candidate identity");
  }
  const {data,error}=await (client||getSupabaseClient()).rpc(
    "apply_candidate_memory_observation",{
      p_queue_item_id:queueItemId,
      p_candidate_id:id,
      p_candidate_key:candidateKey,
      p_action:action,
      p_skill_id:skillId,
      p_category:category,
      p_candidate:candidate,
      p_metadata:metadata,
      p_expected_occurrences:expectedOccurrences
    });
  if(error)throw new Error(error.message||"Candidate observation atomic write failed");
  if(!data||!data.candidate||typeof data.candidate!=="object")
    throw new Error("Candidate observation RPC returned no persisted Candidate");
  const row=data.candidate;
  if(!row.id||typeof row.candidate_json!=="object"||
     (data.applied!==true&&data.existing!==true))
    throw new Error("Invalid Candidate observation response");
  return {
    applied:data.applied===true,
    existing:data.existing===true,
    memory:{
      id:row.id,candidateKey:row.candidate_key,
      action:row.action,skillId:row.skill_id,
      category:row.category,status:row.status,
      confidence:Number(row.confidence||0),
      occurrences:Number(row.occurrences||0),
      candidate:row.candidate_json,
      metadata:row.metadata||{},
      createdAt:row.created_at||null,
      updatedAt:row.updated_at||null,
      lastSeenAt:row.last_seen_at||null
    }
  };
}
