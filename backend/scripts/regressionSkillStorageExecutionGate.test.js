import test from "node:test";
import assert from "node:assert/strict";
import {loadExperiences} from "../experience/storage/supabaseExperienceStore.js";
import {resolveExperience} from "../experience/experienceCore.js";
import {mergeResolvedSubtaskExperience}
  from "../core/subtask/mergeResolvedSubtaskExperience.js";
import {buildExecutionExperienceContext}
  from "../core/subtask/subtaskExperienceContext.js";
import {facade,rows} from "./fixtures/v2ReadbackSupabaseSnapshot.js";

const skillId="disposable-safety-v2";
const latest={...facade,id:skillId,version:4,name:"Newest Skill"};
const old={id:skillId,version:3,name:"Previous Skill",
  moduleManifest:{version:1,counts:{workflow:1}}};
const memory={source:"memory-context",skills:[{
  id:skillId,name:"stale ready skill",
  learning:{maturityLevel:"READY"},
  workflow:["unsafe cached routine"]
}]};

function mockedSupabase({corrupt=false,moduleError=false}={}){
  const reads=[];
  const skills=[
    {id:skillId,version:3,payload:old,enabled:true,status:"published"},
    {id:skillId,version:4,payload:latest,enabled:true,status:"published"}
  ];
  const queryData=({skill_id,version})=>{
    reads.push([skill_id,version]);
    if(moduleError)return {data:null,error:{message:"injected Supabase read failure"}};
    if(version===3)return {data:[{
      module_key:"workflow",part_no:0,data:["stale old workflow"]
    }],error:null};
    const data=structuredClone(rows);
    if(corrupt)data[2].data=["corrupted same-length output"];
    return {data,error:null};
  };
  const client={from(table){
    if(table==="jessica_experience_skills"){
      const q={
        select(){return q;},eq(){return q;},
        order(){return Promise.resolve({data:structuredClone(skills),error:null});}
      };
      return q;
    }
    assert.equal(table,"jessica_experience_skill_modules");
    const filters={};
    const q={
      select(){return q;},
      eq(field,value){filters[field]=value;return q;},
      order(field){
        if(field==="part_no")return Promise.resolve(queryData(filters));
        return q;
      }
    };
    return q;
  }};
  return {client,reads};
}

async function resolvedFromStore(state){
  return resolveExperience("Evaluate selected skill",
    null,()=>loadExperiences(state.client));
}

for(const testCase of [
  {name:"corrupt latest module",state:{corrupt:true}},
  {name:"Supabase module read error",state:{moduleError:true}}
]){
  test("failed Skill hydration never crosses execution boundary: "+testCase.name,async()=>{
    const mock=mockedSupabase(testCase.state);
    const found=await resolvedFromStore(mock);
    assert.equal(found.found,false);
    assert.equal(found.reason,"storage-error");
    assert.equal(found.experience,null);
    const merged=mergeResolvedSubtaskExperience(found,memory);
    const exec=buildExecutionExperienceContext(merged);
    assert.equal(merged.found,false);
    assert.equal(merged.experience,null);
    assert.deepEqual(merged.planningContext.plannerHints,[]);
    assert.deepEqual(merged.planningContext.constraints,[]);
    assert.deepEqual(exec.skills,[]);
    assert.equal(exec.found,false);
    assert.equal(exec.used,false);
    assert.equal(exec.context.experience,null);
    assert.deepEqual(mock.reads,[[skillId,4]],
      "corrupt latest Skill must not be replaced by an earlier version");
  });
}

test("non-storage no-match keeps the established memory fallback",()=>{
  const resolved={found:false,reason:"no-match",experience:null};
  const merged=mergeResolvedSubtaskExperience(resolved,memory);
  assert.equal(merged.source,"memory-context");
  assert.equal(merged.found,true);
  assert.equal(merged.experience,memory);
});

test("valid full Skill remains preferred to stale memory hints",async()=>{
  const state=mockedSupabase();
  const skills=await loadExperiences(state.client);
  const selected={found:true,experience:skills[0]};
  const merged=mergeResolvedSubtaskExperience(selected,memory);
  const execution=buildExecutionExperienceContext(merged);
  assert.equal(execution.used,true);
  assert.equal(execution.skills.length,1);
  assert.equal(execution.skills[0].version,4);
  assert.deepEqual(execution.skills[0].workflow,
    ["read source","verify output"]);
  assert.deepEqual(state.reads,[[skillId,4]]);
});
