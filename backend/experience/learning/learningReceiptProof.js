import {validateExperienceReceipt}
  from "../storage/experiencePersistence/experienceReceiptIdentity.js";

// A successful lookup alone does not prove the returned Skill belongs to this
// Proposal. Never promote a Candidate on a malformed or mismatched receipt.
export function verifyExistingLearningReceipt({
  receipt,proposalId,skillId
}={}){
  const proof=validateExperienceReceipt({
    saved:receipt,proposalId,skillId
  });
  return proof.valid===true
    ? {valid:true,receipt:proof.history}
    : {valid:false,reason:proof.reason||"Skill receipt proof unavailable"};
}
