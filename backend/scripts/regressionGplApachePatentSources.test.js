import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan,tryCreateKnownLicenseComparison} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
import {validateResult} from "../core/validator.js";
const question="Сравни патентные условия GPLv3 и Apache 2.0 по официальным текстам. Укажи патентные лицензии, основания прекращения прав, пункты и ссылки.";
const gpl=[
"GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.",
"8. Termination. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11).",
"10. Automatic Licensing of Downstream Recipients. You may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by the Program.",
"11. Patents. Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims, to make, use, sell, offer for sale, import and otherwise run, modify and propagate the contents of its contributor version.",
"END OF TERMS AND CONDITIONS"
].join(" ");
const apache=[
"Apache License Version 2.0, January 2004. TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION.",
"3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s).",
"If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
"4. Redistribution. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work.",
"If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file.",
"END OF TERMS AND CONDITIONS"
].join(" ");
const plan=buildKnownLicenseComparisonPlan(question);
const run=(a=gpl,b=apache)=>({success:true,results:plan.steps.map((x,i)=>({
tool:"web_fetch",success:true,data:{url:x.arguments.url,content:i===0?a:b,truncated:false}
}))});
test("GPLv3 and Apache 2.0 use both official sources without an AI composer",async()=>{
 assert.equal(tryCreateKnownLicenseComparison({task:question}).handled,true);
 assert.deepEqual(plan.steps.map(x=>x.arguments.url),[
 "https://www.gnu.org/licenses/gpl-3.0.txt","https://www.apache.org/licenses/LICENSE-2.0.txt"]);
 const answer=groundedPatentComparison(question,plan,run());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.match(answer.text,/раздел 11/);
 assert.match(answer.text,/Раздел 8/);
 assert.match(answer.text,/Раздел 10/);
 assert.match(answer.text,/раздел 3/);
 assert.doesNotMatch(answer.text,/невозможно|не подтвержден/i);
 assert.equal((await validateResult(question,plan,run(),answer)).valid,true);
});
test("wrong official content, truncation, or reversed lawsuit clause fails closed",()=>{
 assert.equal(groundedPatentComparison(question,plan,run(gpl.replace("essential patent claims","some patent claims"),apache)),null);
 assert.equal(groundedPatentComparison(question,plan,run(gpl,apache.replace("including a cross-claim","excluding a cross-claim"))),null);
 const broken=run();broken.results[0].data.truncated=true;
 assert.equal(groundedPatentComparison(question,plan,broken),null);
});
