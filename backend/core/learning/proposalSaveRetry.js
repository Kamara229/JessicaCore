import {getSupabaseClient} from "../../storage/supabaseClient.js";

const MAX_ATTEMPTS=3;

// Database row-lock and journal proof decide whether it is safe to retry.
// This API never retries the Proposal insert itself and cannot update Skills.
export async function recordProposalSaveFailure({
  queueItemId,maxAttempts=MAX_ATTEMPTS,client=null
}={}){
  if(typeof queueItemId!=="string"||!queueItemId.trim())
    return {success:false,status:"invalid",error:"Queue item ID required"};
  try{
    const limit=Number.isInteger(maxAttempts)
      ?Math.max(1,Math.min(5,maxAttempts)):MAX_ATTEMPTS;
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "record_learning_proposal_save_failure",{
        p_queue_item_id:queueItemId,p_max_attempts:limit
      });
    if(error)return {success:false,status:"rpc-error",
      error:error.message||String(error)};
    const status=String(data?.status||"");
    if(!["retry","exhausted","already_saved","unverified",
          "not_processing","not_found"].includes(status))
      return {success:false,status:"rpc-error",error:"Unknown Proposal retry result"};
    const attempts=Number(data?.attempts||0);
    if(!Number.isSafeInteger(attempts)||attempts<0||attempts>limit)
      return {success:false,status:"rpc-error",error:"Invalid Proposal retry count"};
    return {
      success:["retry","exhausted","already_saved"].includes(status),
      status,attempts,limit,
      proposalId:data?.proposal_id||null,
      queueStatus:data?.queue_status||null
    };
  }catch(error){
    return {success:false,status:"rpc-error",
      error:error?.message||"Proposal retry RPC unavailable"};
  }
}
