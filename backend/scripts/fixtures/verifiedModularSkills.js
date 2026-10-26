import {facade as v2Snapshot,rows as v2SnapshotRows}
  from "./v2ReadbackSupabaseSnapshot.js";

// Stable, sandbox-only records; mixed versions use independent pinned IDs.
export const v1Id="ci-verified-skill-v1";
export const v2Id="ci-verified-skill-v2";
export const v1Payload={
  id:v1Id,version:3,name:"Verified v1 Skill",
  moduleManifest:{version:1,counts:{workflow:1}},
  learning:{maturityLevel:"READY",occurrences:3,confidence:0.92}
};
export const v2Payload={
  ...v2Snapshot,id:v2Id,version:4,name:"Verified v2 Skill",
  learning:{maturityLevel:"READY",occurrences:3,confidence:0.92}
};
export const manifestRows=[
  {id:v1Id,version:3,enabled:true,status:"published",payload:v1Payload},
  {id:v2Id,version:4,enabled:true,status:"published",payload:v2Payload}
];
export const moduleRows=new Map([
  [v1Id+":3",[{module_key:"workflow",part_no:0,
    data:["read source","verify output"]}]],
  [v2Id+":4",v2SnapshotRows]
]);
export function modularExperienceClient(){
  const query=(table)=>{
    const filters={};
    const api={
      select(){return api;},
      eq(key,value){filters[key]=value;return api;},
      order(key){
        if(table==="jessica_experience_skills")return Promise.resolve({
          data:structuredClone(manifestRows),error:null});
        if(key==="part_no")return Promise.resolve({
          data:structuredClone(moduleRows.get(filters.skill_id+":"+filters.version)),
          error:null});
        return api;
      }
    };
    return api;
  };
  return {from:query};
}
