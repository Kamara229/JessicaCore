import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task="Сравни патентные условия Apache 2.0 и MIT License по официальным текстам. Укажи ссылки и пункты.";
const apache=[
 "Apache License Version 2.0. TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION.",
 "3. Grant of Patent License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s).",
 "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
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


test("live MIT/Apache task states precise patent-claim termination trigger",()=>{
 const live="Сравни патентные условия MIT License и Apache 2.0 по официальным текстам. Есть ли явная патентная лицензия и когда она прекращается? Укажи пункты и ссылки.";
 const livePlan=buildKnownLicenseComparisonPlan(live);
 const out=groundedPatentComparison(live,livePlan,run());
 assert.equal(out?.source,"bounded-patent-comparison");
 assert.match(out.text,/раздел 3/);
 assert.match(out.text,/со дня подачи иска/);
 assert.match(out.text,/встречного или перекрёстного иска/);
 assert.match(out.text,/MIT не имеет нумерованных/);
 assert.match(out.text,/подразумеваемых патентных прав/);
 assert.ok(out.text.includes("https://www.apache.org/licenses/LICENSE-2.0.txt"));
 assert.ok(out.text.includes("https://opensource.org/license/mit"));
 assert.equal(groundedPatentComparison(live,livePlan,
     run(apache.replace("as of the date such litigation is filed","after a court ruling"),mit)),null);
 assert.equal(groundedPatentComparison(live,livePlan,
     run(apache.replace("including a cross-claim","excluding a cross-claim"),mit)),null);
});

test("exact ordinary-chat Apache MIT request returns a sourced table with short literal quotes",()=>{
 const request="Сравни Apache License 2.0 и MIT по патентным правам и их прекращению. Составь таблицу с условиями, короткими цитатами и официальными ссылками";
 const planned=buildKnownLicenseComparisonPlan(request);
 const answer=groundedPatentComparison(request,planned,run());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.ok(answer.text.startsWith("| Критерий | Apache License 2.0 | MIT License |"));
 assert.ok(answer.text.includes("| Патентная лицензия |"));
 assert.ok(answer.text.includes("| Прекращение |"));
 assert.match(answer.text,/«patent license to make, have made, use»/u);
 assert.match(answer.text,/«shall terminate as of the date such litigation is filed»/u);
 assert.match(answer.text,/«Permission is hereby granted, free of charge»/u);
 assert.ok(answer.text.includes("https://www.apache.org/licenses/LICENSE-2.0.txt"));
 assert.ok(answer.text.includes("https://opensource.org/license/mit"));
});
