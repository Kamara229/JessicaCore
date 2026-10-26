import {createHash,randomUUID} from "node:crypto";

// Stable across retries and replays; different Skills in one Trace stay distinct.
export function createLearningQueueIdentity(event,skillId=null){
  const traceId=String(event?.traceId||event?.id||"").trim();
  const action=String(event?.action||"").trim().toUpperCase();
  const target=String(event?.skillId||skillId||"").trim();
  if(!traceId||!action)return randomUUID(); // legacy event compatibility
  const hash=createHash("sha256")
    .update(JSON.stringify([traceId,action,target]),"utf8")
    .digest("hex");
  return "jessica-lq-v1-"+hash;
}
