import test from "node:test";
import assert from "node:assert/strict";
import {packageExperienceModules} from "../experience/storage/skillModules/experienceModulePacker.js";
import {hydrateExperienceModules} from "../experience/storage/skillModules/experienceModuleHydrator.js";

const sample={
 id:"test-modular-skill",version:3,
 name:"Verifier",keywords:["official","source"],
 workflow:["fetch source","verify author","cite exact excerpt"],
 validationRules:["validate source URL","do not infer absence"],
 constraints:["no unsupported facts"],
 examples:Array.from({length:7},(_,i)=>({task:"Verify "+i,result:{
  text:"documented excerpt "+String(i).repeat(650),
  success:true},success:true,traceId:"trace-"+i}))
};
function mockClient(rows){
 return {from:()=>({select:()=>({eq:()=>({eq:()=>({
   order:()=>({order:async()=>({data:rows,error:null})})
 })})})})};
}
test("split large Skills into bounded small function-scoped modules",()=>{
 const {facade,modules}=packageExperienceModules(sample);
 assert.equal(facade.examples,undefined);
 assert.equal(facade.workflow,undefined);
 assert.deepEqual(facade.keywords,sample.keywords);
 assert.ok(modules.length>=5);
 assert.ok(modules.every(x=>Buffer.byteLength(JSON.stringify(x.data))<=4096));
 assert.ok(facade.moduleManifest?.counts.examples>=2);
});
test("version-pinned skill facade reconstructs original learning fields",async()=>{
 const {facade,modules}=packageExperienceModules(sample);
 const rows=modules.map(x=>({
    module_key:x.moduleKey,part_no:x.part,data:x.data
 }));
 const hydrated=await hydrateExperienceModules(facade,sample.id,sample.version,mockClient(rows));
 for(const field of ["workflow","validationRules","constraints","examples"])
   assert.deepEqual(hydrated[field],sample[field]);
 assert.deepEqual(hydrated.keywords,sample.keywords);
 assert.equal(hydrated.moduleManifest,undefined);
});
test("missing Skill modules fail closed",async()=>{
 const {facade,modules}=packageExperienceModules(sample);
 const rows=modules.slice(1).map(x=>({module_key:x.moduleKey,part_no:x.part,data:x.data}));
 await assert.rejects(
  hydrateExperienceModules(facade,sample.id,sample.version,mockClient(rows)),
  /integrity/
 );
});
test("large single array item is rejected rather than silently cut",()=>{
 assert.throws(()=>packageExperienceModules({...sample,
  examples:[{task:"x".repeat(5000)}]}),/Oversize/);
});
test("backward compatibility: legacy unsharded payload is untouched",async()=>{
 const full=await hydrateExperienceModules(sample,sample.id,sample.version,mockClient([]));
 assert.equal(full,sample);
});
