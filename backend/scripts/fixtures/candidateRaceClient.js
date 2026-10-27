// A deterministic PostgREST row adapter: an UPDATE checks all predicates
// at commit, as a single PostgreSQL statement would do.
const ID="00000000-0000-4000-8000-000000000611";
const REV="2026-10-20T12:00:00.000Z";
export const candidateRaceIdentity={id:ID,revision:REV};

export function candidateRaceClient(){
  let row={
    id:ID,status:"ACTIVE",updated_at:REV,occurrences:1,
    candidate_json:{examples:[{traceId:"original"}],occurrences:1},
    metadata:{source:"verified"}
  };
  let writes=0,hold=null;
  function holdNextUpdate(kind){
    if(hold)throw Error("Only one race hook may be active");
    let entered,release;
    const arrived=new Promise(resolve=>{entered=resolve;});
    const gate=new Promise(resolve=>{release=resolve;});
    hold={kind,arrived,entered,gate,release};
    return {entered:arrived,release};
  }
  const client={from(table){
    if(table!=="learning_candidates")throw Error("Unexpected table");
    const predicates=[],query={patch:null,
      select(){return this;},
      update(patch){this.patch=patch;return this;},
      eq(column,value){predicates.push([column,value]);return this;},
      async maybeSingle(){
        if(this.patch===null)
          return {data:predicates.every(([k,v])=>row[k]===v)?
            structuredClone(row):null,error:null};
        const kind=this.patch.status?"finalize":"evidence";
        if(hold?.kind===kind){
          const current=hold;
          hold=null;
          current.entered();
          await current.gate;
        }
        if(!predicates.every(([k,v])=>row[k]===v))
          return {data:null,error:null};
        row={...row,...this.patch};
        writes++;
        return {data:structuredClone(row),error:null};
      }
    };
    return query;
  }};
  return {client,holdNextUpdate,
    get row(){return structuredClone(row);},
    get writes(){return writes;}
  };
}
