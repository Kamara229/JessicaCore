import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan,tryCreateKnownLicenseComparison} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
const task="Сравни патентные условия GPLv3 и MPL 2.0 по официальным текстам. Укажи ссылки, пункты и условия прекращения прав.";
const gpl="GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. "+
"8. Termination. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11). However, if you cease all violation of this License, then your license is reinstated. "+
"10. Automatic Licensing of Downstream Recipients. You may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed. "+
"11. Patents. Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims. END OF TERMS AND CONDITIONS";
const mpl="Mozilla Public License Version 2.0. "+
"2.1. Grants. Under Patent Claims of such Contributor to make, use, sell. "+
"2.3. Limitations on Grant Scope. "+
"5.1. The rights granted under this License will terminate automatically if You fail to comply. "+
"5.2. If You initiate litigation against any entity by asserting a patent infringement claim (excluding declaratory judgment actions, counter-claims, and cross-claims), the rights granted under Section 2.1 of this License shall terminate.";
const plan=buildKnownLicenseComparisonPlan(task);
const results=(g=gpl,m=mpl)=>({results:plan.steps.map((s,i)=>({tool:"web_fetch",success:true,data:{url:s.arguments.url,content:i===0?g:m,truncated:false}}))});
test("GPLv3/MPL official-source answer includes both termination regimes",()=>{
 assert.equal(tryCreateKnownLicenseComparison({task}).handled,true,"GPLv3 alias must not be rejected by grounding");
 assert.deepEqual(plan.steps.map(s=>s.arguments.url),["https://www.gnu.org/licenses/gpl-3.0.txt","https://www.mozilla.org/media/MPL/2.0/index.txt"]);
 const output=groundedPatentComparison(task,plan,results());
 assert.equal(output?.source,"bounded-patent-comparison");
 assert.match(output.text,/раздел 11/);
 assert.match(output.text,/Раздел 10/);
 assert.match(output.text,/5\.2/);
});
test("changed litigation exception or incomplete source cannot be verified",()=>{
 assert.equal(groundedPatentComparison(task,plan,results(gpl.replace("including a cross-claim","excluding a cross-claim"),mpl)),null);
 assert.equal(groundedPatentComparison(task,plan,results(gpl,mpl.replace("excluding declaratory","including declaratory"))),null);
 const broken=results();broken.results[0].data.truncated=true;
 assert.equal(groundedPatentComparison(task,plan,broken),null);
});
