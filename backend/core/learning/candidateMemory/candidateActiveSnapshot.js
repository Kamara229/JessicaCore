// A database-owned revision is incremented by trigger on EVERY Candidate
// UPDATE. Unlike millisecond timestamps, revision cannot collide in a race.
export async function writeActiveCandidateSnapshot({
  client,id,expectedUpdatedAt,expectedRevision,payload
}={}){
  if(!client||!id||typeof expectedUpdatedAt!=="string"||
     !expectedUpdatedAt.trim()||
     !Number.isSafeInteger(expectedRevision)||expectedRevision<0||
     !payload||typeof payload!=="object"||Array.isArray(payload))
    throw new Error("ACTIVE Candidate revision proof required");
  const {data,error}=await client.from("learning_candidates")
    .update(payload).eq("id",id).eq("status","ACTIVE")
    .eq("revision",expectedRevision)
    .eq("updated_at",expectedUpdatedAt).select("*").maybeSingle();
  if(error)throw new Error(error.message||"Candidate update unavailable");
  if(!data)throw new Error("Candidate changed or finalized; update aborted");
  if(data.status!=="ACTIVE"||data.revision!==expectedRevision+1)
    throw new Error("Candidate revision was not advanced atomically");
  return data;
}
