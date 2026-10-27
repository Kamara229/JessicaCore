import {getSupabaseClient} from "../../../storage/supabaseClient.js";
import {validatedCandidateOwnerReceipt} from "./approvalOwnerReceipt.js";

// Read-only: the unique DB claim identifies the winning Proposal.
// Never infer a loser from a transient RPC failure alone.
export async function readCandidatePublicationOwner(candidateId,{client=null}={}){
  const db=client||getSupabaseClient();
  const {data:claim,error:claimError}=await db
    .from("jessica_candidate_skill_publications")
    .select("candidate_id,proposal_id,skill_id,skill_version")
    .eq("candidate_id",candidateId).maybeSingle();
  if(claimError)throw Error(claimError.message||"Publication claim unavailable");
  if(claim)return validatedCandidateOwnerReceipt(candidateId,claim);
  // For Skills published before the claim table existed, consult PROMOTED
  // ownership; ACTIVE with no claim is not evidence of a lost race.
  const {data:candidate,error}=await db.from("learning_candidates")
    .select("status,metadata").eq("id",candidateId).maybeSingle();
  if(error)throw Error(error.message||"Candidate ownership unavailable");
  if(candidate?.status==="PROMOTED" &&
     candidate.metadata?.proposalId)
    return {proposalId:candidate.metadata.proposalId,
      skillId:candidate.metadata.skillId||null,
      evidence:"legacy-promoted-owner"};
  return null;
}
