import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task="Сравни патентные условия MPL 2.0 и MIT License по официальным текстам. Укажи ссылки, пункты и прекращение прав.\nВерни ответ дословно.";
const mpl=[
"Mozilla Public License Version 2.0.",
"2.1. Grants. Each Contributor hereby grants You a world-wide, royalty-free, non-exclusive license: under intellectual property rights to use and reproduce the work; under Patent Claims of such Contributor to make, use, sell, offer for sale, have made, import, and otherwise transfer either its Contributions or its Contributor Version.",
"2.3. Limitations on Grant Scope. The licenses granted in this Section 2 are the only rights granted under this License. No additional rights or licenses will be implied from the distribution or licensing of Covered Software under this License. Notwithstanding Section 2.1(b) above, no patent license is granted by a Contributor for any code that a Contributor has removed from Covered Software, for infringements caused by third-party modifications, or for combinations except as part of the Contributor Version.",
"5.1. The rights granted under this License will terminate automatically if You fail to comply with any of its terms. However, if You become compliant the rights granted under this License are reinstated under the detailed conditions in Section 5.1.",
"5.2. If You initiate litigation against any entity by asserting a patent infringement claim (excluding declaratory judgment actions, counter-claims, and cross-claims) alleging that a Contributor Version directly or indirectly infringes any patent, then the rights granted to You by any and all Contributors for the Covered Software under Section 2.1 of this License shall terminate."
].join(" ");
const mit=[
"The MIT License. Copyright <YEAR> <COPYRIGHT HOLDER>",
"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:",
"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
"THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.",
"IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join(" ");
const plan=buildKnownLicenseComparisonPlan(task);
const run=(m=mpl,t=mit)=>({results:plan.steps.map((step,index)=>({
 tool:"web_fetch",success:true,data:{url:step.arguments.url,content:index===0?m:t,truncated:false}
}))});
test("MPL 2.0 and MIT compare directly from official primary texts",()=>{
 assert.equal(plan?.intent,"native_known_license_comparison");
 assert.deepEqual(plan.steps.map(x=>x.arguments.url),[
  "https://www.mozilla.org/media/MPL/2.0/index.txt",
  "https://opensource.org/license/mit"
 ]);
 const answer=groundedPatentComparison(task,plan,run());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.match(answer.text,/2\.1\(b\)/);
 assert.match(answer.text,/5\.2/);
 assert.match(answer.text,/MIT не имеет нумерованных пунктов/);
 assert.match(answer.text,/подразумеваемых патентных прав/);
});
test("missing operative patent litigation limits or MIT disclaimer fail closed",()=>{
 assert.equal(groundedPatentComparison(task,plan,run(mpl.replace("excluding declaratory judgment actions, counter-claims, and cross-claims","including counterclaims"),mit)),null);
 assert.equal(groundedPatentComparison(task,plan,run(mpl,mit.replace("OTHER DEALINGS IN THE SOFTWARE","OTHER DEALINGS WITH THE CODE"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(mpl,mit.replace("THE SOFTWARE IS PROVIDED AS IS","A patent license is hereby granted. THE SOFTWARE IS PROVIDED AS IS"))),null);
});
