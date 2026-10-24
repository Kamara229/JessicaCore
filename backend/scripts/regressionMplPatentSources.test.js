import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {verifyPairedSourceContent} from "../core/validator/pairedSourceContent.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const question="Сравни патентные условия MPL 2.0 и Apache 2.0 по официальным текстам. Укажи ссылки, пункты и условия прекращения патентных прав.";
const urls=["https://www.apache.org/licenses/LICENSE-2.0.txt","https://www.mozilla.org/en-US/MPL/2.0/"];
const apache=[
 "Apache License Version 2.0, January 2004. TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION.",
 "3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted.",
 "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
 "4. Redistribution. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work.",
 "If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file."
].join(" ");
const mozilla=[
 "Mozilla Public License Version 2.0.",
 "2.1. Grants. Each Contributor hereby grants You a world-wide, royalty-free, non-exclusive license: under intellectual property rights to use and reproduce the work; under Patent Claims of such Contributor to make, use, sell, offer for sale, have made, import, and otherwise transfer either its Contributions or its Contributor Version.",
 "2.3. Limitations on Grant Scope. The licenses granted in this Section 2 are the only rights granted under this License. No additional rights or licenses will be implied from the distribution or licensing of Covered Software under this License. Notwithstanding Section 2.1(b) above, no patent license is granted by a Contributor for any code that a Contributor has removed from Covered Software, for infringements caused by third-party modifications, or for combinations except as part of the Contributor Version.",
 "5.1. The rights granted under this License will terminate automatically if You fail to comply with any of its terms. However, if You become compliant the rights granted under this License are reinstated under the detailed conditions in Section 5.1.",
 "5.2. If You initiate litigation against any entity by asserting a patent infringement claim (excluding declaratory judgment actions, counter-claims, and cross-claims) alleging that a Contributor Version directly or indirectly infringes any patent, then the rights granted to You by any and all Contributors for the Covered Software under Section 2.1 of this License shall terminate."
].join(" ");
const plan=buildKnownLicenseComparisonPlan(question);
const result=(one=apache,two=mozilla)=>({results:plan.steps.map((s,i)=>({
 tool:"web_fetch",success:true,data:{url:s.arguments.url,content:i===0?one:two,truncated:false}
}))});
test("patent MPL/Apache routed to Mozilla and Apache primary texts, never SPDX",()=>{
 assert.ok(plan);
 assert.equal(plan.intent,"native_known_license_comparison");
 assert.deepEqual(plan.steps.map(s=>s.arguments.url),urls);
});
test("source-grounded answer compares 5.2 cross claims exception with Apache section 3",()=>{
 assert.equal(verifyPairedSourceContent(plan,result()).valid,true);
 const answer=groundedPatentComparison(question,plan,result());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.match(answer.text,/5\.2/);
 assert.match(answer.text,/5\.1/);
 assert.match(answer.text,/встречные и перекрёстные/);
 assert.match(answer.text,/mozilla.org\/en-US\/MPL\/2\.0/);
 assert.match(answer.text,/apache.org/);
});
test("missing Mozilla official text or crucial carveout must fail closed",()=>{
 assert.equal(groundedPatentComparison(question,plan,result(apache,"SPDX.org summary")),null);
 assert.equal(groundedPatentComparison(question,plan,result(apache,
   mozilla.replace("excluding declaratory judgment actions, counter-claims, and cross-claims","including all cross-claims"))),null);
});
