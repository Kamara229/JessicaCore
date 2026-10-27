import {validateExperienceReceipt} from "./experienceReceiptIdentity.js";

// A lost atomic-save ACK is verified through immutable Proposal history only.
// Never retry the write and never convert an unverified commit into failure.
export async function recoverExperienceSaveReceipt({
  proposalId,skillId,reason,lookup
}={}){
  const failure=String(reason||"Experience save unconfirmed");
  if(!proposalId||typeof lookup!=="function")
    return {success:false,existing:false,uncertain:true,error:failure};
  try{
    const saved=await lookup(proposalId);
    if(!saved)
      return {success:false,existing:false,uncertain:true,error:failure};
    const proof=validateExperienceReceipt({saved,proposalId,skillId});
    if(!proof.valid)
      return {success:false,existing:false,uncertain:true,error:proof.reason};
    return {success:true,existing:true,commitRecovered:true,
      skillId:proof.skillId,version:proof.version,
      experience:proof.experience,history:proof.history,
      storage:"commit-ack-readback"};
  }catch(error){
    return {success:false,existing:false,uncertain:true,
      error:"Experience receipt lookup unavailable: "+
        (error?.message||failure)};
  }
}
