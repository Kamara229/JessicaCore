import test from "node:test";
import assert from "node:assert/strict";
import {loadExperiences} from "../experience/storage/supabaseExperienceStore.js";
import {facade as v2Facade,rows as v2Modules}
 from "./fixtures/v2ReadbackSupabaseSnapshot.js";

const v2Id="mixed-v2-active";
const v1Id="mixed-v1-active";
const legacy={
  id:v1Id,version:3,name:"Legacy module",moduleManifest:{
    version:1,counts:{workflow:1}
  }
};
const modern={...v2Facade,id:v2Id,version:4,name:"Verified v2"};
const inactive={id:"inactive-corrupt",moduleManifest:{
  version:2,counts:{workflow:1},sha256:{workflow:["0".repeat(64)]}
}};
const row=(id,version,payload,enabled=true,status="published")=>({
  id,version,payload,enabled,status
});
const stored=[
  row(v2Id,1,{...inactive,id:v2Id},true),
  row(v1Id,3,legacy),
  row("disabled",100,inactive,false,"disabled"),
  row(v2Id,4,modern),
  row("draft",100,inactive,true,"draft")
];

function mockStore(skillRows=stored, corruptV2=false){
  const observed={skillFilters:[],moduleReads:[]};
  const modulesByVersion=new Map([
    [v1Id+":3",[{module_key:"workflow",part_no:0,data:["legacy step"]}]],
    [v2Id+":4",structuredClone(v2Modules)]
  ]);
  if(corruptV2){
    modulesByVersion.get(v2Id+":4")[2].data=["tampered output"];
  }
  const client={from(table){
    if(table==="jessica_experience_skills"){
      const q={
        select(fields){assert.match(fields,/payload/);return q;},
        eq(k,v){observed.skillFilters.push([k,v]);return q;},
        order(k){assert.equal(k,"version");
          return Promise.resolve({data:structuredClone(skillRows),error:null});}
      };
      return q;
    }
    assert.equal(table,"jessica_experience_skill_modules");
    const filters={};
    const q={
      select(fields){assert.equal(fields,"module_key,part_no,data");return q;},
      eq(k,v){filters[k]=v;return q;},
      order(k){if(k!=="part_no")return q;
        observed.moduleReads.push([filters.skill_id,filters.version]);
        return Promise.resolve({data:structuredClone(
          modulesByVersion.get(filters.skill_id+":"+filters.version)||[]
        ),error:null});}
    };
    return q;
  }};
  return {client,observed};
}

test("mixed active v1/v2 Skill loading restores pinned modules in sequence",async()=>{
  const {client,observed}=mockStore();
  const actual=await loadExperiences(client);
  assert.equal(actual.length,2);
  const byId=new Map(actual.map(s=>[s.id,s]));
  assert.equal(byId.get(v1Id).version,3);
  assert.deepEqual(byId.get(v1Id).workflow,["legacy step"]);
  assert.equal(byId.get(v2Id).version,4);
  assert.deepEqual(byId.get(v2Id).workflow,["read source","verify output"]);
  assert.deepEqual(byId.get(v2Id).examples,
    [{task:"checksum roundtrip",status:"ok"}]);
  assert.equal(byId.get(v1Id).moduleManifest,undefined);
  assert.equal(byId.get(v2Id).moduleManifest,undefined);
  assert.deepEqual(observed.skillFilters,
    [["enabled",true],["status","published"]]);
  assert.deepEqual(observed.moduleReads,[[v1Id,3],[v2Id,4]]);
});

test("higher version wins even if Supabase results are unordered",async()=>{
  const {client,observed}=mockStore([...stored].reverse());
  const found=await loadExperiences(client);
  assert.equal(found.find(s=>s.id===v2Id).version,4);
  assert.deepEqual(observed.moduleReads,[[v2Id,4],[v1Id,3]]);
});

test("corrupt latest v2 Skill cannot be returned or replaced by older version",async()=>{
  const {client,observed}=mockStore(stored,true);
  await assert.rejects(loadExperiences(client),/Skill module checksum mismatch/);
  assert.deepEqual(observed.moduleReads,[[v1Id,3],[v2Id,4]]);
});

test("disabled and draft Skills cannot enter hydration or active selection",async()=>{
  const {client,observed}=mockStore(stored.filter(r=>r.enabled!==true||r.status!=="published"));
  assert.deepEqual(await loadExperiences(client),[]);
  assert.deepEqual(observed.moduleReads,[]);
});

test("same id/version conflict and invalid active version fail closed",async()=>{
  const duplicate=[row(v1Id,3,legacy),row(v1Id,3,legacy)];
  await assert.rejects(loadExperiences(mockStore(duplicate).client),
    /Conflicting active Skill versions/);
  await assert.rejects(loadExperiences(mockStore([row(v1Id,0,legacy)]).client),
    /invalid id or version/);
});
