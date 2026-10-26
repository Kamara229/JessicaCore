import test from "node:test";
import assert from "node:assert/strict";
import {packageExperienceModules}
 from "../experience/storage/skillModules/experienceModulePacker.js";
import {hydrateExperienceModules}
 from "../experience/storage/skillModules/experienceModuleHydrator.js";
import {hashExperienceModulePart}
 from "../experience/storage/skillModules/experienceModuleIntegrity.js";

const original={id:"signed-module-regression",version:4,name:"Checksum",
 workflow:["read","validate"],examples:[
  {task:"a",result:{status:"ok",details:{z:3,a:1}}},
  {task:"b",result:{status:"ok",details:{b:2,c:4}}}
 ]};
const rows=modules=>modules.map(item=>({
 module_key:item.moduleKey,part_no:item.part,data:structuredClone(item.data)
}));
const client=parts=>({from:()=>({select:()=>({eq:()=>({eq:()=>({
 order:()=>({order:async()=>({data:parts,error:null})})
})})})})});

test("v2 manifest hashes recover original modules despite JSONB key ordering",async()=>{
 const {facade,modules}=packageExperienceModules(original);
 assert.equal(facade.moduleManifest.version,2);
 const originalPart=modules.find(m=>m.moduleKey==="examples");
 const reordered=structuredClone(originalPart.data);
 reordered[0].result={details:{a:1,z:3},status:"ok"};
 assert.equal(hashExperienceModulePart(reordered),
   hashExperienceModulePart(originalPart.data));
 const all=rows(modules);
 all.find(x=>x.module_key==="examples").data[0]=reordered[0];
 const restored=await hydrateExperienceModules(facade,original.id,4,client(all));
 assert.deepEqual(restored.examples,original.examples);
 assert.deepEqual(restored.workflow,original.workflow);
});

test("same-length, wrong content cannot hydrate immutable v2 Skill",async()=>{
 const {facade,modules}=packageExperienceModules(original);
 const corrupt=rows(modules);
 const row=corrupt.find(x=>x.module_key==="examples");
 row.data[0].task="x";
 await assert.rejects(
  hydrateExperienceModules(facade,original.id,4,client(corrupt)),
  /checksum mismatch/
 );
 const invalid=structuredClone(facade);
 invalid.moduleManifest.sha256.examples[0]="0".repeat(64);
 await assert.rejects(
  hydrateExperienceModules(invalid,original.id,4,client(rows(modules))),
  /checksum mismatch/
 );
});

test("legacy immutable v1 modules remain readable without rewriting history",async()=>{
 const {facade,modules}=packageExperienceModules(original);
 const old=structuredClone(facade);
 old.moduleManifest={version:1,counts:facade.moduleManifest.counts};
 const loaded=await hydrateExperienceModules(old,original.id,4,client(rows(modules)));
 assert.deepEqual(loaded.examples,original.examples);
});
