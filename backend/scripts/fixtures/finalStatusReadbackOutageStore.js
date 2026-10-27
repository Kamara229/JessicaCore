// Fault-injection store: the DB is durable; only transport ACK/readback fails.
// It models a fresh worker using the same committed tables after a restart.
export function finalStatusReadbackOutageStore({
  postCommitAckLoss=false,readbackOutage=true,updateOutage=false
}={}){
  const state={
    row:{id:"outage-proposal",status:"PROCESSING",approved_at:null,
      processing_at:"2026-10-27T10:00:00Z",rejected_at:null},
    candidate:{status:"PROMOTED",revision:1,
      proposalId:"outage-proposal",skillId:"outage-skill"},
    versions:new Map([["outage-proposal",1]]),
    statusWrites:0,versionWrites:0,candidateWrites:0,
    readbackFailures:0,updateAttempts:0
  };
  let readbackDown=readbackOutage;
  let updateDown=updateOutage;
  let loseAck=postCommitAckLoss;

  const client={from(table){
    if(table!=="learning_proposals")
      throw Error("Unexpected table: "+table);
    const q={payload:null,filters:{},
      update(payload){this.payload=payload;return this;},
      select(){return this;},
      eq(key,value){this.filters[key]=value;return this;},
      async maybeSingle(){
        const matched=Object.entries(this.filters)
          .every(([key,value])=>state.row[key]===value);
        if(this.payload===null){
          if(readbackDown){
            state.readbackFailures++;
            return {data:null,error:{message:"readback unavailable"}};
          }
          return {data:matched?structuredClone(state.row):null,error:null};
        }
        state.updateAttempts++;
        if(updateDown)
          return {data:null,error:{message:"update transport unavailable"}};
        if(!matched)return {data:null,error:null};
        state.row={...state.row,...this.payload};
        state.statusWrites++;
        if(loseAck){
          loseAck=false;
          return {data:null,error:{message:"status committed; ACK lost"}};
        }
        return {data:structuredClone(state.row),error:null};
      }
    };
    return q;
  }};

  return {state,client,
    setReadbackAvailable(available){readbackDown=!available;},
    setUpdateAvailable(available){updateDown=!available;},
    get row(){return structuredClone(state.row);},
    createWorker(){
      return {
        approve:async({proposal})=>{
          const version=state.versions.get(proposal.id);
          if(!version)throw Error("Fixture must start after Skill publication");
          return {success:true,existing:true,stage:"already-saved",
            skillId:"outage-skill",version,
            experience:{id:"outage-skill",version}};
        },
        promote:async({proposal,skillId})=>{
          if(state.candidate.status!=="PROMOTED"||
              state.candidate.proposalId!==proposal.id||
              state.candidate.skillId!==skillId)
            return {success:false,error:"Candidate ownership conflict"};
          return {success:true,existing:true,updated:false};
        }
      };
    }
  };
}
