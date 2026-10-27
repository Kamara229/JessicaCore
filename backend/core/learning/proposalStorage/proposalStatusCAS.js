// Every Proposal status transition uses an atomic source-state predicate.
// A terminal Proposal can only be confirmed again, never overwritten.
const TABLE="learning_proposals";
const TERMINAL=new Set(["APPROVED","REJECTED","FAILED","KEEP_CANDIDATE"]);
const ALLOWED={
  APPROVED:"PROCESSING",REJECTED:"PROCESSING",
  FAILED:"PROCESSING",KEEP_CANDIDATE:"PROCESSING",
  PROCESSING:"PENDING_APPROVAL",PENDING_APPROVAL:"PROCESSING"
};

async function readStatus(client,id){
  const {data,error}=await client.from(TABLE).select("*")
    .eq("id",id).maybeSingle();
  if(error)throw Error(error.message||"Proposal readback unavailable");
  return data;
}

export async function transitionProposalStatus({
  client,id,status,payload
}={}){
  if(!client||!id||!ALLOWED[status]||!payload||
     payload.status!==status)
    return {success:false,error:"Invalid Proposal status transition"};
  try{
    // Atomic UPDATE must not overwrite terminal or non-owned Proposal.
    const {data,error}=await client.from(TABLE).update(payload)
      .eq("id",id).eq("status",ALLOWED[status])
      .select("*").maybeSingle();
    if(data?.status===status&&!error)
      return {success:true,existing:false,updated:true,proposal:data};
    // No row or lost ACK: check for a committed exact status.
    const stored=await readStatus(client,id);
    if(stored?.status===status)
      return {success:true,existing:true,updated:false,
        recoveredAck:!!error,proposal:stored};
    if(error)return {success:false,uncertain:true,
      error:error.message||"Proposal transition ACK unavailable"};
    return {success:false,conflict:true,
      error:stored?"Proposal status conflict":"Proposal not found",
      storedStatus:stored?.status||null};
  }catch(error){
    return {success:false,uncertain:true,
      error:error?.message||"Proposal status storage unavailable"};
  }
}
