import {getSupabaseClient} from "../../storage/supabaseClient.js";

const DEFAULT_STALE_MINUTES=120;
const DEFAULT_LIMIT=100;

// Only a service-role Supabase RPC may change stale leases. Fail closed if
// the RPC is absent: no client-side UPDATE/SELECT race or automatic retry
// of ambiguous Candidate Memory evidence.
export async function recoverStaleLearningQueue({
  staleMinutes=process.env.LEARNING_QUEUE_STALE_MINUTES,
  limit=DEFAULT_LIMIT,
  client=null
}={}){
  const mins=Number(staleMinutes);
  const requested=Number(limit);
  const p_stale_minutes=Number.isFinite(mins)
    ? Math.max(120,Math.min(10080,Math.floor(mins)))
    : DEFAULT_STALE_MINUTES;
  const p_limit=Number.isFinite(requested)
    ? Math.max(1,Math.min(100,Math.floor(requested)))
    : DEFAULT_LIMIT;
  try{
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "recover_stale_learning_queue",{p_stale_minutes,p_limit});
    if(error)return {success:false,error:error.message||String(error),recovered:0};
    if(!Array.isArray(data))
      return {success:false,error:"Invalid stale queue recovery response",recovered:0};
    if(data.some(row=>!["PENDING","PROPOSED","FAILED"].includes(row?.status)))
      return {success:false,error:"Unexpected recovery transition",recovered:0};
    return {
      success:true,recovered:data.length,
      retried:data.filter(x=>x.status==="PENDING").length,
      provenComplete:data.filter(x=>x.status==="PROPOSED").length,
      quarantined:data.filter(x=>x.status==="FAILED").length,
      entries:data.map(({id,action,status})=>({id,action,status}))
    };
  }catch(error){
    return {success:false,error:error?.message||"Queue recovery unavailable",recovered:0};
  }
}
