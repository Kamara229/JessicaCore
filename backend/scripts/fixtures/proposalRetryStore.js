// In-memory transactional outcome model for service-only Proposal retry RPC.
// Mirrors the SQL state machine; real Supabase rollback is tested separately.
export function proposalRetryStore(queueItemId,{journal=true,limit=3}={}){
  const queue={id:queueItemId,status:"PROCESSING",action:"NEW_SKILL"};
  const proposals=new Map();
  let attempts=0,calls=0;
  const client={async rpc(name,args){
    if(name!=="record_learning_proposal_save_failure")
      throw new Error("Unexpected RPC "+name);
    if(args.p_queue_item_id!==queueItemId)throw new Error("Wrong queue item");
    calls++;
    if(queue.status!=="PROCESSING")
      return {data:{status:"not_processing",attempts:0},error:null};
    if(proposals.has(queueItemId)){
      queue.status="PROPOSED";
      return {data:{status:"already_saved",attempts:0,
        proposal_id:proposals.get(queueItemId).id},error:null};
    }
    if(!journal)return {data:{status:"unverified",attempts:0},error:null};
    attempts++;
    queue.status=attempts>=limit?"FAILED":"PENDING";
    return {data:{status:queue.status==="FAILED"?"exhausted":"retry",
      attempts,limit,queue_status:queue.status},error:null};
  }};
  return {
    client,queue,proposals,
    get attempts(){return attempts;},
    get calls(){return calls;},
    reclaim(){if(queue.status!=="PENDING")throw Error("Not claimable");
      queue.status="PROCESSING";}
  };
}
