// Resolve a lost atomic-save ACK only through immutable Proposal history.
// Never reissue the write or claim success from an unconfirmed RPC result.
export async function recoverExperienceSaveReceipt({
  proposalId,skillId,reason,lookup
}={}){
  const failure=String(reason||"Experience save unconfirmed");
  if(!proposalId||typeof lookup!=="function")
    return {success:false,existing:false,uncertain:!!proposalId,error:failure};
  try{
    const saved=await lookup(proposalId);
    if(!saved)
      return {success:false,existing:false,uncertain:true,error:failure};
    const id=saved.skillId||saved.payload?.id;
    const version=Number(saved.version);
    const owner=saved.payload?.metadata?.proposalId;
    if(id!==skillId||owner!==proposalId||
       !Number.isSafeInteger(version)||version<1)
      return {success:false,existing:false,uncertain:true,
        error:"Experience receipt conflicts with requested Proposal"};
    return {success:true,existing:true,commitRecovered:true,
      skillId:id,version,experience:saved.payload,
      history:saved,storage:"commit-ack-readback"};
  }catch(error){
    return {success:false,existing:false,uncertain:true,
      error:"Experience receipt lookup unavailable: "+
        (error?.message||failure)};
  }
}
