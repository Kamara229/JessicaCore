// A small ownership gate for Pattern Discovery before and after AI extraction.
// Read-only checkpoint; persistent mutations remain fenced by database RPCs.
export async function checkPatternDiscoveryClaim(item,verifyClaim){
  if(item?.action!=="PATTERN_DISCOVERY")return {valid:true,stale:false};
  try{
    const result=await verifyClaim({
      queueItemId:item.id,
      claimToken:item.claimToken||item.claim_token||null
    });
    if(result?.valid===true)return {valid:true,stale:false};
    return {
      valid:false,
      stale:result?.stale===true,
      reason:result?.reason||"Pattern Discovery claim not verified"
    };
  }catch(error){
    return {
      valid:false,stale:false,
      reason:error?.message||"Pattern Discovery claim check unavailable"
    };
  }
}
