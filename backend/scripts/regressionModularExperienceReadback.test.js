import test from "node:test";
import assert from "node:assert/strict";
import {hydrateExperienceModules} from "../experience/storage/skillModules/experienceModuleHydrator.js";
import {facade,rows,skillId,version} from "./fixtures/v2ReadbackSupabaseSnapshot.js";

function fakeSupabase(parts,observed){
  const query={
    select(columns){assert.equal(columns,"module_key,part_no,data");return this;},
    eq(field,value){observed.push([field,value]);return this;},
    order(field,opts){observed.push([field,opts?.ascending]);return this;},
    then(resolve,reject){return Promise.resolve({data:structuredClone(parts),error:null})
      .then(resolve,reject);}
  };
  return {from(table){
    assert.equal(table,"jessica_experience_skill_modules");
    return query;
  }};
}

const read=async(parts=rows)=>{
  const observed=[];
  const restored=await hydrateExperienceModules(
    structuredClone(facade),skillId,version,fakeSupabase(parts,observed));
  assert.deepEqual(observed,[
    ["skill_id",skillId],["version",version],
    ["module_key",true],["part_no",true]]);
  return restored;
};

test("actual v2 Supabase readback restores the ordered facade and removes manifest",async()=>{
  const restored=await read();
  assert.equal(restored.moduleManifest,undefined);
  assert.deepEqual(restored.workflow,["read source","verify output"]);
  assert.deepEqual(restored.examples,[{task:"checksum roundtrip",status:"ok"}]);
  assert.equal(restored.version,version);
  assert.equal(restored.id,skillId);
  assert.equal(facade.moduleManifest.version,2,"input must remain unmodified");
});

test("v2 module readback rejects same-length corruption",async()=>{
  const corrupt=structuredClone(rows);
  corrupt[2].data=["wrong output"];
  await assert.rejects(read(corrupt),/checksum mismatch/);
});

test("v2 module readback rejects exchanged part payloads",async()=>{
  const wrong=structuredClone(rows);
  [wrong[1].data,wrong[2].data]=[wrong[2].data,wrong[1].data];
  await assert.rejects(read(wrong),/checksum mismatch/);
});

test("v2 module readback rejects sequence gaps, omissions and extras",async()=>{
  const gap=structuredClone(rows);
  gap[2].part_no=3;
  await assert.rejects(read(gap),/sequence corrupted/);
  await assert.rejects(read(rows.slice(1)),/integrity failure/);
  await assert.rejects(read([...rows,rows[2]]),/integrity failure/);
});

test("v2 module readback rejects checksum mismatch in manifest",async()=>{
  const changed=structuredClone(facade);
  changed.moduleManifest.sha256.workflow[0]="0".repeat(64);
  await assert.rejects(hydrateExperienceModules(changed,skillId,version,
    fakeSupabase(rows,[])),/checksum mismatch/);
});
