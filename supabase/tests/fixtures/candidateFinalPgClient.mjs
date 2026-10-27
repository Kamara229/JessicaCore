import {sql} from "./candidateRacePgClient.mjs";

// Adapter for the real Candidate revision/updated_at compare-and-swap facade.
export function candidateFinalPgClient(id,{readbackDown=false}={}){
  return {from(table){
    if(table!=="learning_candidates")throw Error("Unexpected Candidate table");
    return {
      payload:null,filters:{},
      select(){return this;},
      update(value){this.payload=value;return this;},
      eq(key,value){this.filters[key]=value;return this;},
      async maybeSingle(){
        if(this.filters.id!==id)throw Error("Candidate identity mismatch");
        if(this.payload===null){
          const r=await sql(`
SELECT row_to_json(t)::text FROM (
 SELECT id,status,metadata,revision,updated_at FROM public.learning_candidates
 WHERE id='${id}') t;
`,"jessica-ci-candidate-read",readbackDown?{PGPORT:"1"}:{});
          return r.code===0
            ? {data:r.stdout?JSON.parse(r.stdout):null,error:null}
            : {data:null,error:{message:"Candidate readback unavailable"}};
        }
        if(this.payload.status!=="PROMOTED"||
           this.filters.status!=="ACTIVE"||
           !Number.isSafeInteger(this.filters.revision))
          throw Error("Invalid Candidate CAS probe");
        const metadata=JSON.stringify(this.payload.metadata).replaceAll("'","''");
        const expected=String(this.filters.updated_at).replaceAll("'","''");
        const r=await sql(`
UPDATE public.learning_candidates SET status='PROMOTED',
 metadata='${metadata}'::jsonb,
 updated_at=now()
WHERE id='${id}' AND status='ACTIVE'
 AND revision=${this.filters.revision}
 AND updated_at='${expected}'::timestamptz
RETURNING row_to_json(learning_candidates)::text;
`,"jessica-ci-candidate-promote");
        return r.code===0
          ? {data:r.stdout?JSON.parse(r.stdout):null,error:null}
          : {data:null,error:{message:r.stderr}};
      }
    };
  }};
}
