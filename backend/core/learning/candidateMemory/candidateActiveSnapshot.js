// CAS update: only a still-ACTIVE Candidate snapshot may be modified.
// A concurrent approval (PROMOTED/REJECTED) or merge (MERGED) wins cleanly.
export async function writeActiveCandidateSnapshot({
  client,id,expectedUpdatedAt,payload
}={}){
  if(!client||!id||typeof expectedUpdatedAt!=="string"||
     !expectedUpdatedAt.trim()||!payload||
     typeof payload!=="object"||Array.isArray(payload))
    throw new Error("ACTIVE Candidate snapshot proof required");
  const {data,error}=await client.from("learning_candidates")
    .update(payload).eq("id",id).eq("status","ACTIVE")
    .eq("updated_at",expectedUpdatedAt).select("*").maybeSingle();
  if(error)throw new Error(error.message||"Candidate update unavailable");
  if(!data)throw new Error("Candidate changed or finalized; update aborted");
  if(data.status!=="ACTIVE")
    throw new Error("Candidate mutation returned non-ACTIVE row");
  return data;
}
