import {getSupabaseClient} from "../../../storage/supabaseClient.js";
import {buildProposalInsertPayload,normalizeDatabaseProposal}
  from "./proposalStorageMapper.js";

// Queue lock and proposal insert happen within one SQL RPC. A stale Worker
// cannot persist a Proposal after the Queue claim token was rotated.
export async function saveLearningProposalOwned(
  proposal,{claimToken,client=null}={}
){
  if(!proposal?.queueItemId||!claimToken)
    return {success:false,error:"Proposal requires current Queue claim token"};
  try{
    const payload=buildProposalInsertPayload(proposal);
    const {data,error}=await (client||getSupabaseClient()).rpc(
      "save_learning_proposal_owned",{
        p_queue_item_id:proposal.queueItemId,
        p_claim_token:claimToken,
        p_payload:payload
      });
    if(error)return {success:false,error:error.message||String(error)};
    if(data?.saved!==true || !data.proposal?.id)
      return {success:false,stale:data?.reason==="stale_claim",
        error:data?.reason||"Proposal ownership rejected"};
    return {success:true,existing:data.existing===true,
      proposal:normalizeDatabaseProposal(data.proposal)};
  }catch(error){
    return {success:false,error:error?.message||"Owned Proposal write failed"};
  }
}
