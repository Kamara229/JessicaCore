// A claim is evidence only when its returned identity and version agree
// with the Candidate requested by the caller. Unknown is not a loss.
export function validatedCandidateOwnerReceipt(candidateId,claim){
  if(!claim)return null;
  const expected=String(candidateId||"").toLowerCase();
  const actual=typeof claim.candidate_id==="string"
    ? claim.candidate_id.toLowerCase():"";
  const proposalId=claim.proposal_id;
  const skillId=claim.skill_id;
  const version=claim.skill_version;
  if(!expected||actual!==expected||
    typeof proposalId!=="string"||!proposalId.trim()||
    typeof skillId!=="string"||!skillId.trim()||
    !Number.isSafeInteger(version)||version<=0)
    throw Error("Candidate publication owner receipt is inconsistent");
  return {
    proposalId,skillId,
    evidence:"atomic-publication-claim"
  };
}
