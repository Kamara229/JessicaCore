import {readCandidatePublicationOwner} from "./approvalPublicationClaimReader.js";

function candidateId(proposal){
  return proposal?.candidateMemory?.id||
    proposal?.analysis?.candidateMemory?.id||
    proposal?.provenance?.candidateMemoryId||
    proposal?.analysis?.provenance?.candidateMemoryId||null;
}

// A retry becomes terminal only when durable ownership proves a different
// Proposal already won the same Candidate. Errors always remain uncertain.
export async function checkLosingCandidatePublication({
  proposal,readOwner=readCandidatePublicationOwner
}={}){
  const id=candidateId(proposal);
  if(proposal?.action!=="NEW_SKILL"||!proposal?.id||!id)
    return {confirmed:false,checked:false};
  try{
    const owner=await readOwner(id);
    if(owner?.proposalId&&owner.proposalId!==proposal.id)
      return {confirmed:true,checked:true,candidateId:id,
        winningProposalId:owner.proposalId,evidence:owner.evidence,
        reason:"Candidate already published by another Proposal"};
    return {confirmed:false,checked:true,candidateId:id};
  }catch(error){
    return {confirmed:false,checked:false,uncertain:true,candidateId:id,
      error:error?.message||"Candidate publication ownership unavailable"};
  }
}
