import {getSupabaseClient} from "../../storage/supabaseClient.js";

// A Worker can only extend its own non-expired claim. The SQL row lock
// serializes renewal with stale recovery and token rotation.
export async function renewLearningQueueClaim({
  queueItemId,claimToken,client=null
}={}){
  if(!queueItemId||!claimToken)
    return {renewed:false,stale:true,reason:"Claim owner token missing"};
  try{
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "heartbeat_learning_queue_claim",{
        p_queue_item_id:queueItemId,p_claim_token:claimToken
      });
    if(error)return {renewed:false,stale:false,
      reason:error.message||String(error)};
    if(data?.renewed!==true)return {
      renewed:false,stale:data?.reason==="claim_expired_or_stale",
      reason:data?.reason||"Heartbeat unverified"
    };
    return {renewed:true,stale:false,processedAt:data.processed_at||null};
  }catch(error){
    return {renewed:false,stale:false,
      reason:error?.message||"Heartbeat unavailable"};
  }
}
