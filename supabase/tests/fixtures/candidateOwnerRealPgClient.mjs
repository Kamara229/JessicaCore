import {assertIsolatedCi,sql} from "./candidateRacePgClient.mjs";

// PostgreSQL-backed PostgREST-shaped read adapter, scoped to disposable CI.
export function candidateOwnerRealPgClient(candidateId,{tamper=null}={}){
  assertIsolatedCi();
  if(!/^[a-f\d-]{36}$/iu.test(candidateId))throw Error("invalid CI Candidate ID");
  return {from(table){
    if(!["jessica_candidate_skill_publications","learning_candidates"].includes(table))
      throw Error("unexpected readback table");
    return {
      select(){return this;},
      eq(key,value){
        if(key!==(table==="learning_candidates"?"id":"candidate_id")||
           value!==candidateId)throw Error("unexpected Candidate identity");
        return this;
      },
      async maybeSingle(){
        const q=table==="learning_candidates"
          ?`SELECT row_to_json(t)::text FROM
             (SELECT status,metadata FROM public.learning_candidates
              WHERE id='${candidateId}') t;`
          :`SELECT row_to_json(t)::text FROM
             (SELECT candidate_id,proposal_id,skill_id,skill_version
              FROM public.jessica_candidate_skill_publications
              WHERE candidate_id='${candidateId}') t;`;
        const result=await sql(q,"jessica-ci-owner-receipt-readback");
        if(result.code!==0)return {data:null,error:{message:result.stderr}};
        const row=result.stdout?JSON.parse(result.stdout):null;
        if(row&&table==="jessica_candidate_skill_publications"){
          if(tamper==="wrong-candidate")
            row.candidate_id="00000000-0000-4000-8000-000000000999";
          if(tamper==="missing-skill")row.skill_id=null;
          if(tamper==="invalid-version")row.skill_version=0;
        }
        return {data:row,error:null};
      }
    };
  }};
}
