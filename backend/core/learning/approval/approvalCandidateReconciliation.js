// The Skill has committed, but the Proposal must not become terminal
// until its linked Candidate is confirmed PROMOTED (or absent by design).
import {promoteCandidateMemory} from "./approvalCandidateMemory.js";

export async function reconcileCandidatePromotion({
  proposal,skillId,promote=promoteCandidateMemory
}={}){
  try{
    const result=await promote({proposal,skillId});
    if(result?.success===true)return result;
    return {...(result||{}),success:false,updated:false,
      error:result?.error||"Candidate PROMOTED not confirmed"};
  }catch(error){
    return {success:false,updated:false,skipped:false,
      error:error?.message||"Candidate PROMOTED unavailable"};
  }
}
