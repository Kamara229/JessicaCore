import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task="Сравни патентные условия Apache 2.0 и MIT License по официальным текстам. Укажи ссылки и пункты.";
const apache=[
 "Apache License Version 2.0. TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION.",
 "3. Grant of Patent License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s).",
 "If You institute patent litigation against any entity alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
 "4. Redistribution. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices. If the Work includes a NOTICE text file, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file."
].join(" ");
const mit=[
 "The MIT License. Copyright <YEAR> <COPYRIGHT HOLDER>",
 "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:",
 "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
 "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.",
 "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join(" ");
const plan=buildKnownLicenseComparisonPlan(task);
const run=(a=apache,b=mit)=>({results:plan.steps.map((step,i)=>({
 tool:"web_fetch",success:true,data:{url:step.arguments.url,content:i===0?a:b,truncated:false}
}))});
test("Apache/MIT patent clauses are explained only from validated canonical texts",()=>{
 assert.deepEqual(plan.steps.map(x=>x.tool),["web_fetch","web_fetch"]);
 const out=groundedPatentComparison(task,plan,run());
 assert.equal(out?.source,"bounded-patent-comparison");
 assert.match(out.text,/раздел 3/);
 assert.match(out.text,/MIT не имеет нумерованных|MIT не содержит нумерованных/);
 assert.match(out.text,/подразумеваемых патентных прав/);
 assert.match(out.text,/apache.org/);
 assert.match(out.text,/opensource.org/);
});
test("incomplete or contradictory MIT operative text fails closed",()=>{
 assert.equal(groundedPatentComparison(task,plan,run(apache,mit.replace("OTHER DEALINGS IN THE SOFTWARE","OTHER ACTIONS"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(apache,mit.replace("THE SOFTWARE IS PROVIDED AS IS","A patent clause is hereby granted. THE SOFTWARE IS PROVIDED AS IS"))),null);
});
