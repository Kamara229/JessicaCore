import {getSupabaseClient} from "../../storage/supabaseClient.js";

// Service-only SQL RPC uses FOR UPDATE SKIP LOCKED; never substitute an
// unprotected SELECT when claiming fails.
export async function claimPendingLearningItems({limit=100,client=null}={}){
  try{
    const safeLimit=Math.max(1,Math.min(100,Number.isInteger(limit)?limit:100));
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "claim_pending_learning_queue",{p_limit:safeLimit});
    if(error)return {success:false,items:[],error:error.message||String(error)};
    if(!Array.isArray(data))
      return {success:false,items:[],error:"Invalid atomic claim result"};
    if(data.some(row=>typeof row?.claim_token!=="string" ||
         !/^[0-9a-f]{8}-[0-9a-f-]{27,}$/i.test(row.claim_token))){
      return {success:false,items:[],error:"Atomic claim missing owner token"};
    }
    return {success:true,count:data.length,items:data.map(row=>({
      ...row,
      skillId:row.skill_id??null,
      event:row.event_json||{},
      createdAt:row.created_at??null,
      processedAt:row.processed_at??null,
      claimToken:row.claim_token??null
    }))};
  }catch(error){
    return {success:false,items:[],error:error?.message||"Atomic claim unavailable"};
  }
}
