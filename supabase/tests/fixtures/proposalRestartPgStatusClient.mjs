import {sql} from "./candidateRacePgClient.mjs";

// Minimal PostgREST-like adapter; every UPDATE and SELECT is real PostgreSQL.
export function proposalStatusClient(phase){
  return {from(table){
    if(table!=="learning_proposals")throw Error("Unexpected table");
    return {
      payload:null,filters:{},
      update(payload){this.payload=payload;return this;},
      select(){return this;},
      eq(key,value){this.filters[key]=value;return this;},
      async maybeSingle(){
        const {id,status}=this.filters;
        if(id!=="ci-restarted-proposal")throw Error("Unexpected Proposal ID");
        if(this.payload){
          if(this.payload.status!=="APPROVED"||status!=="PROCESSING")
            throw Error("Invalid status CAS test");
          const approved=String(this.payload.approved_at).replaceAll("'","''");
          const result=await sql(`
UPDATE public.learning_proposals SET status='APPROVED',
 approved_at='${approved}'::timestamptz, processing_at=NULL,
 rejected_at=NULL WHERE id='ci-restarted-proposal' AND status='PROCESSING'
RETURNING row_to_json(learning_proposals)::text;
`,"jessica-proposal-status-update");
          if(result.code!==0)return {data:null,error:{message:result.stderr}};
          if(result.stdout&&phase==="lost-ack")
            return {data:null,error:{message:"Proposal COMMIT confirmed; ACK dropped"}};
          return {data:result.stdout?JSON.parse(result.stdout):null,error:null};
        }
        const result=await sql(`
SELECT row_to_json(t)::text FROM (
 SELECT id,status,approved_at,processing_at,rejected_at
 FROM public.learning_proposals WHERE id='ci-restarted-proposal'
) t;
`,"jessica-proposal-readback",
          phase==="restored"?{}:{PGPORT:"1"});
        return result.code===0
          ? {data:result.stdout?JSON.parse(result.stdout):null,error:null}
          : {data:null,error:{message:"Proposal readback unavailable"}};
      }
    };
  }};
}
