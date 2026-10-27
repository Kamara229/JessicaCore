import {randomUUID} from "node:crypto";

// Small deterministic PostgREST mock: each RPC represents one SQL transaction.
export function ownerTokenQueueClient(){
  const row={id:"lease-item",action:"NEW_SKILL",status:"PENDING",
    claim_token:null,processed_at:null,event_json:{}};
  const proposals=new Map();
  let retries=0;
  const tokenMatches=(token)=>
    row.status==="PROCESSING"&&row.claim_token===token;
  const client={async rpc(name,args){
    if(name==="claim_pending_learning_queue"){
      if(row.status!=="PENDING")return {data:[],error:null};
      row.status="PROCESSING";
      row.claim_token=randomUUID();
      row.processed_at=new Date().toISOString();
      return {data:[structuredClone(row)],error:null};
    }
    if(name==="transition_learning_queue_owned"){
      if(!tokenMatches(args.p_claim_token))
        return {data:{updated:false,reason:"stale_claim"},error:null};
      row.status=args.p_target_status;
      row.claim_token=null;
      return {data:{updated:true,status:row.status},error:null};
    }
    if(name==="record_learning_proposal_save_failure_owned"){
      if(!tokenMatches(args.p_claim_token))
        return {data:{status:"stale_claim",attempts:0},error:null};
      retries++;
      row.status=retries>=args.p_max_attempts?"FAILED":"PENDING";
      row.claim_token=null;
      return {data:{status:row.status==="FAILED"?"exhausted":"retry",
        attempts:retries,limit:args.p_max_attempts},error:null};
    }
    if(name==="save_learning_proposal_owned"){
      if(!tokenMatches(args.p_claim_token))
        return {data:{saved:false,reason:"stale_claim"},error:null};
      let item=proposals.get(args.p_queue_item_id);
      const existing=Boolean(item);
      if(!item){
        item=structuredClone(args.p_payload);
        proposals.set(args.p_queue_item_id,item);
      }
      return {data:{saved:true,existing,proposal:structuredClone(item)},error:null};
    }
    throw Error("Unexpected RPC "+name);
  }};
  return {
    client,row,proposals,
    get retries(){return retries;},
    recover(){
      if(row.status!=="PROCESSING")throw Error("Not processing");
      row.status="PENDING";row.claim_token=null;row.processed_at=null;
    }
  };
}
