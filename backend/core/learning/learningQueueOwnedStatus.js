import {getSupabaseClient} from "../../storage/supabaseClient.js";

// Strict lease fencing; no unguarded fallback after a claim UUID exists.
export async function transitionLearningQueueOwned({
  queueItemId,claimToken,status,client=null
}={}){
  if(!queueItemId||!claimToken||
     !["PENDING","PROPOSED","IGNORED","FAILED"].includes(status)){
    return {success:false,error:"Current queue ownership token required"};
  }
  try{
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "transition_learning_queue_owned",{
        p_queue_item_id:queueItemId,
        p_claim_token:claimToken,
        p_target_status:status
      });
    if(error)return {success:false,error:error.message||String(error)};
    if(data?.updated!==true || data?.status!==status){
      return {success:false,stale:data?.reason==="stale_claim",
        error:data?.reason||"Queue ownership rejected"};
    }
    return {success:true,status:data.status};
  }catch(error){
    return {success:false,error:error?.message||"Queue transition unavailable"};
  }
}
