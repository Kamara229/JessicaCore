import {randomUUID} from "node:crypto";

// Atomic Supabase RPC emulator for one Queue lease and its Candidate journal.
export function ownedCandidateJournalClient(){
  const row={id:"candidate-fence-queue",status:"PENDING",claim_token:null};
  const journal=new Map(),candidates=new Map();
  let writes=0;
  const matches=token=>row.status==="PROCESSING"&&row.claim_token===token;
  const client={async rpc(name,p){
    if(name==="claim_pending_learning_queue"){
      if(row.status!=="PENDING")return {data:[],error:null};
      row.status="PROCESSING";row.claim_token=randomUUID();
      return {data:[{...row,action:"NEW_SKILL",event_json:{}}],error:null};
    }
    if(name!=="apply_candidate_memory_observation_owned")
      return {data:null,error:{message:"Legacy unfenced observation rejected"}};
    if(!matches(p.p_claim_token))
      return {data:null,error:{message:"Stale Candidate Memory Queue ownership"}};
    if(journal.has(p.p_queue_item_id)){
      return {data:{applied:false,existing:true,
        candidate:structuredClone(candidates.get(journal.get(p.p_queue_item_id)))},
        error:null};
    }
    const id=p.p_candidate_id,previous=candidates.get(id);
    if(p.p_action==="EXACT"){
      if(!previous||previous.occurrences!==p.p_expected_occurrences)
        return {data:null,error:{message:"Stale candidate observation"}};
    }
    const candidate=structuredClone(p.p_candidate);
    const stored={
      id,candidate_key:p.p_candidate_key,action:"NEW_SKILL",status:"ACTIVE",
      candidate_json:candidate,metadata:p.p_metadata,
      occurrences:Number(candidate.occurrences||1),
      confidence:Number(candidate.confidence||0)
    };
    journal.set(p.p_queue_item_id,id);candidates.set(id,stored);writes++;
    return {data:{applied:true,existing:false,candidate:structuredClone(stored)},
      error:null};
  }};
  return {
    client,row,journal,candidates,
    get writes(){return writes;},
    recover(){if(row.status!=="PROCESSING")throw Error("not processing");
      row.status="PENDING";row.claim_token=null;}
  };
}
