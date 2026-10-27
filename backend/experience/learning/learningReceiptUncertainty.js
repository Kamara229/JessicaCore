// Read outages cannot prove whether a Proposal's Skill was committed.
// Keep the persistent Proposal PROCESSING until receipt/history can be read.
export function deferLearningReceiptVerification({
  proposal,stage,error
}={}){
  return {
    success:false,learned:false,existing:false,
    uncertain:true,deferred:true,stage,
    proposal:proposal||null,experience:null,
    error:error?.message||String(error||"Experience receipt verification unavailable")
  };
}
