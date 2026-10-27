import {getSupabaseClient} from "../../storage/supabaseClient.js";

// A cheap read-only checkpoint after costly asynchronous Pattern Discovery.
// Only transactional owned Candidate/Proposal writers authorize mutations.
export async function verifyLearningQueueClaim({
  queueItemId,claimToken,client=null
}={}){
  if(!queueItemId||!claimToken){
    return {valid:false,stale:true,reason:"Queue ownership token required"};
  }
  try{
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "is_learning_queue_claim_current",{
        p_queue_item_id:queueItemId,p_claim_token:claimToken
      });
    if(error)return {valid:false,stale:false,reason:error.message||String(error)};
    if(data!==true)return {
      valid:false,stale:data===false,
      reason:data===false?"Queue claim expired":"Ownership check returned no proof"
    };
    return {valid:true,stale:false};
  }catch(error){
    return {valid:false,stale:false,reason:error?.message||"Ownership check failed"};
  }
}
