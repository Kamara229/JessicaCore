import {randomUUID} from "node:crypto";

// Mock atomic heartbeat RPC; token rotation represents stale recovery + reclaim.
export function patternHeartbeatLeaseClient(){
  const row={id:"heartbeat-item",status:"PROCESSING",
    claim_token:randomUUID(),processed_at:Date.now()};
  let calls=0;
  const client={async rpc(name,p){
    if(name!=="heartbeat_learning_queue_claim")
      throw new Error("Unexpected RPC "+name);
    calls++;
    if(row.id!==p.p_queue_item_id||row.status!=="PROCESSING"
      ||row.claim_token!==p.p_claim_token){
      return {data:{renewed:false,reason:"claim_expired_or_stale"},error:null};
    }
    row.processed_at=Date.now();
    return {data:{renewed:true,processed_at:new Date(row.processed_at).toISOString()},
      error:null};
  }};
  return {
    client,row,
    get calls(){return calls;},
    rotate(){row.claim_token=randomUUID();return row.claim_token;},
    expire(){row.status="FAILED";row.claim_token=null;},
    get item(){return {id:row.id,action:"PATTERN_DISCOVERY",
      claimToken:row.claim_token};}
  };
}
