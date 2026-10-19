import test from "node:test";
import assert from "node:assert/strict";
import {planComparisonEvidenceRoutes} from "../core/validator/claim/comparisonEvidenceRouting.js";
import {planComparisonBatches} from "../core/validator/claim/comparisonEvidenceController.js";

const titles=["Использование","Распространение","Уведомления","Патентные условия","Ограничения"];
const checks=titles.flatMap((criterion,i)=>[0,1].map(source=>({
 id:1+2*i+source,criterion,source,value:"Checked provision"
})));
const index=checks.filter(x=>x.criterion!=="Патентные условия"||x.source===0).map(x=>({
 id:"E"+x.id,source:x.source,
 criterion:({Использование:"use",Распространение:"distribution",
 Уведомления:"notices","Патентные условия":"patents",Ограничения:"restrictions"})[x.criterion],
 quote:"Verbatim statement of a source provision."
}));

test("one unindexed cell does not disable indexed checking elsewhere",()=>{
 const routes=planComparisonEvidenceRoutes({checks},index);
 assert.ok(routes.some(x=>x.mode==="indexed"));
 assert.ok(routes.some(x=>x.mode==="verbatim"));
 const all=routes.flatMap(x=>x.checks.map(c=>c.id)).sort((a,b)=>a-b);
 assert.deepEqual(all,checks.map(x=>x.id));
 assert.equal(routes.filter(x=>x.mode==="verbatim").flatMap(x=>x.checks).length,1);
});
test("batch allocation is bounded and includes every claim once",()=>{
 const plan=planComparisonBatches({checks},index);
 assert.ok(plan);
 assert.ok(plan.routes.length<=5);
 assert.ok(plan.maxTokens*plan.routes.length<=840);
 assert.ok(plan.routes.every(r=>r.checks.length<=4));
});
test("missing all indexed citations safely selects ordinary mode",()=>{
 const routes=planComparisonEvidenceRoutes({checks},[]);
 assert.ok(routes.every(x=>x.mode==="verbatim"));
});
