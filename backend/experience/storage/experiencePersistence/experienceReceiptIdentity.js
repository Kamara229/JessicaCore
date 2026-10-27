// The durable Proposal receipt is the only proof that a Skill version was saved.
// Match every identity before any caller can report "existing" or "recovered".
export function validateExperienceReceipt({saved,proposalId,skillId}={}){
  if(!saved)return {valid:false,reason:"Experience receipt absent"};
  const payload=saved.payload;
  const version=saved.version;
  if(!payload||typeof payload!=="object"||Array.isArray(payload)||
     !Number.isSafeInteger(version)||version<1)
    return {valid:false,reason:"Invalid immutable Experience receipt"};
  if(!proposalId||!skillId||
     payload.metadata?.proposalId!==proposalId||
     payload.id!==skillId||
     (saved.skillId!=null&&saved.skillId!==skillId))
    return {valid:false,reason:"Experience receipt identity conflict"};
  return {valid:true,skillId,version,experience:payload,history:saved};
}
