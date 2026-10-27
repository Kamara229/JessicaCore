// Supabase recovery must return exactly one integer count within the claim cap.
// Reject malformed or partial responses rather than reporting false success.
export function parseProposalRecoveryCount(data,limit){
  const result=Array.isArray(data)
    ? data.length===1?data[0]:null
    : data;
  const value=result?.recovered;
  if(!Number.isSafeInteger(value)||value<0||value>limit){
    return {valid:false,error:"Invalid Proposal recovery RPC result"};
  }
  return {valid:true,recovered:value};
}
