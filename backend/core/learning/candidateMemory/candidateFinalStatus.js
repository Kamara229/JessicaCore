// Final Candidate transitions are once-only. A replay may confirm the
// same decision, but must never rewrite PROMOTED / REJECTED / MERGED.
const TABLE="learning_candidates";
const FINAL=new Set(["PROMOTED","REJECTED","MERGED"]);
const VOLATILE=new Set(["resolvedAt","mergedAt"]);

function matchingDecision(row,status,metadata){
  if(row?.status!==status)return false;
  const previous=row.metadata||{};
  const keys=Object.keys(metadata).filter(key=>!VOLATILE.has(key));
  // Never accept a keyless replay that could match another Proposal.
  if(!keys.length)return false;
  if(status==="MERGED"&&!metadata.mergedIntoId)return false;
  if(status!=="MERGED"&&!metadata.proposalId)return false;
  return keys.every(key=>previous[key]===metadata[key]);
}

async function read(client,id){
  const {data,error}=await client.from(TABLE).select("*").eq("id",id).maybeSingle();
  if(error)throw new Error(error.message||"Candidate read failed");
  return data;
}

export async function transitionCandidateFinalStatus({
  client,id,status,metadata={}
}={}){
  if(!client||!id||!FINAL.has(status)||
     !metadata||typeof metadata!=="object"||Array.isArray(metadata)||
     (status==="MERGED"?!metadata.mergedIntoId:!metadata.proposalId))
    return {success:false,error:"Invalid Candidate final transition"};
  try{
    const current=await read(client,id);
    if(!current)return {success:false,error:"Candidate not found"};
    if(matchingDecision(current,status,metadata))
      return {success:true,existing:true,updated:false,candidate:current};
    if(current.status!=="ACTIVE")
      return {success:false,conflict:true,error:"Candidate final status conflict"};
    const {data,error}=await client.from(TABLE).update({
      status,metadata:{...(current.metadata||{}),...metadata},
      updated_at:new Date().toISOString()
    }).eq("id",id).eq("status","ACTIVE")
      .eq("updated_at",current.updated_at).select("*").maybeSingle();
    if(error)throw new Error(error.message||"Candidate update failed");
    if(data)return {success:true,existing:false,updated:true,candidate:data};
    // Another Worker may have committed the very same decision concurrently.
    const latest=await read(client,id);
    if(matchingDecision(latest,status,metadata))
      return {success:true,existing:true,updated:false,candidate:latest};
    return {success:false,conflict:true,error:"Candidate transition raced or changed"};
  }catch(error){
    return {success:false,error:error?.message||"Candidate transition unavailable"};
  }
}
