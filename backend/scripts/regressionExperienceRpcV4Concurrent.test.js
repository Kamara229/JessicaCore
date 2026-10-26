import test from "node:test";
import assert from "node:assert/strict";
import {issueExperienceV4Race} from "./fixtures/experienceRpcV4RaceCalls.js";
import {countExperienceRaceRows,deleteExperienceRaceRows}
 from "./fixtures/experienceRpcV4RaceRows.js";

test("two distinct RPC clients issue calls before either finishes",{
  timeout:5000
},async()=>{
  let calls=0,release;
  const gate=new Promise(resolve=>{release=resolve;});
  const factory=()=>({rpc:async(name,args)=>{
    assert.equal(name,"save_jessica_experience_skill_v4");
    assert.equal(args.p_payload.moduleManifest.version,2);
    const seq=++calls;
    if(seq===2)release();
    await gate;
    return {data:{success:true,existing:seq===2,version:1},error:null};
  }});
  const result=await issueExperienceV4Race(factory);
  assert.equal(calls,2);
  assert.equal(result.clientStartSkewMs<1000,true);
  assert.deepEqual(result.outcomes.map(x=>x.status),["fulfilled","fulfilled"]);
  assert.equal(result.outcomes.filter(x=>x.value.existing===false).length,1);
  assert.equal(result.outcomes.filter(x=>x.value.existing===true).length,1);
});

test("actual two-client RPC v4 race preserves one version, modules and audit",{
  skip:!(process.env.SUPABASE_URL&&process.env.SUPABASE_SECRET_KEY),
  timeout:30000
},async()=>{
  const {createClient}=await import("@supabase/supabase-js");
  const factory=()=>createClient(process.env.SUPABASE_URL,
    process.env.SUPABASE_SECRET_KEY,{auth:{autoRefreshToken:false,persistSession:false}});
  const race=await issueExperienceV4Race(factory);
  try{
    assert.ok(race.clientStartSkewMs<1000,"RPC requests did not launch together");
    for(const outcome of race.outcomes){
      assert.equal(outcome.status,"fulfilled",outcome.reason?.message||"RPC failed");
      assert.equal(outcome.value.success,true);
      assert.equal(outcome.value.version,1);
    }
    assert.deepEqual(race.outcomes.map(x=>x.value.existing).sort(),
      [false,true]);
    const counts=await countExperienceRaceRows(race.clients[0],race.sid);
    assert.deepEqual(counts,{skills:1,history:1,modules:1,audit:1,evidence:0});
  }finally{
    await deleteExperienceRaceRows(race.clients[0],race.sid);
  }
});
