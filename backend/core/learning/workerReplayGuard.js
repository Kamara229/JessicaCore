import {findLearningProposalByQueueItemId}
  from "./learningProposalStorage.js";
import {updateLearningQueueItemStatus}
  from "./learningQueueStorage.js";

// Recovery must happen before Candidate Memory or proposal creation.
// Existing proposals retain their status (including APPROVED) on queue replay.
export async function checkLearningWorkerReplay(item,{
  findProposal=findLearningProposalByQueueItemId,
  updateQueue=updateLearningQueueItemStatus
}={}){
  if(!item?.id)return {handled:false};
  const lookup=await findProposal(item.id);
  if(lookup?.success!==true){
    return {handled:true,result:{
      success:true,deferred:true,queueItemId:item.id,
      reason:"Proposal idempotency lookup unavailable; retry later"
    }};
  }
  if(!lookup.found)return {handled:false};
  if(!lookup.proposal?.id){
    return {handled:true,result:{
      success:false,queueItemId:item.id,
      reason:"Existing Proposal lookup returned no identity"
    }};
  }
  const updated=await updateQueue(item.id,"PROPOSED",item.claimToken||null);
  return {handled:true,result:{
    success:updated?.success===true,
    existing:true,
    queueItemId:item.id,
    proposalId:lookup.proposal.id,
    proposal:lookup.proposal,
    queueUpdated:updated?.success===true,
    reason:updated?.success===true
      ? "Existing Proposal recovered without reprocessing"
      : updated?.error||"Queue status update failed"
  }};
}
