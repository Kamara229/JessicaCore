// In-memory PostgREST mock for Supabase INSERT ... ON CONFLICT DO NOTHING.
export function learningQueueInsertOnlyClient(){
  const rows=new Map(),calls=[];
  const client={from(table){
    if(table!=="learning_queue")throw new Error("unexpected table "+table);
    return {
      upsert(payload,options){
        calls.push({id:payload.id,options:{...options}});
        return {
          select(){
            return {
              async maybeSingle(){
                if(rows.has(payload.id)){
                  return {data:null,error:null};
                }
                const row=structuredClone(payload);
                rows.set(payload.id,row);
                return {data:structuredClone(row),error:null};
              }
            };
          }
        };
      }
    };
  }};
  return {client,rows,calls};
}
