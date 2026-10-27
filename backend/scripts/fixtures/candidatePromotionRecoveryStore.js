// Deterministic test DB for Proposal replay when Candidate promotion
// commits but its network acknowledgement is lost.
export function candidatePromotionRecoveryStore(){
  const state={
    proposalStatus:"PROCESSING",
    candidate:{
      id:"00000000-0000-4000-8000-000000000701",
      status:"ACTIVE",revision:0,
      updated_at:"2026-10-27T10:00:00Z",
      metadata:{source:"verified"},occurrences:3
    },
    versions:0,commitAcknowledgmentLost:false,candidateWrites:0,
    proposalWrites:0
  };
  const client={from(table){
    if(table!=="learning_candidates")throw Error("unexpected table");
    const q={payload:null,predicates:{},
      select(){return this;},
      update(value){this.payload=value;return this;},
      eq(key,value){this.predicates[key]=value;return this;},
      async maybeSingle(){
        const matches=Object.entries(this.predicates).every(
          ([k,v])=>state.candidate[k]===v);
        if(!matches)return {data:null,error:null};
        if(!this.payload)return {data:structuredClone(state.candidate),error:null};
        Object.assign(state.candidate,this.payload);
        state.candidate.revision++;
        state.candidateWrites++;
        if(state.commitAcknowledgmentLost){
          state.commitAcknowledgmentLost=false;
          throw Error("Candidate DB ACK lost after COMMIT");
        }
        return {data:structuredClone(state.candidate),error:null};
      }
    };
    return q;
  }};
  return {state,client};
}
