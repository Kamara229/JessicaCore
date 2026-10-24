import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task="Сравни патентные условия Apache 2.0 и BSD 3-Clause по официальным текстам. Приведи ссылки и пункты.";
const apache=[
  "Apache License Version 2.0. TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION.",
  "3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted.",
  "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
  "4. Redistribution. You may reproduce and distribute copies of the Work. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work.",
  "If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file."
].join(" ");
const bsd=[
 "BSD 3-Clause License. Copyright YEAR HOLDER.",
 "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:",
 "1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
 "2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.",
 "3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.",
 "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.",
 "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
].join(" ");

const plan=buildKnownLicenseComparisonPlan(task);
function run(first=apache,second=bsd){
 return {results:plan.steps.map((step,i)=>({
  tool:"web_fetch",success:true,data:{url:step.arguments.url,content:i===0?first:second,truncated:false}
 }))};
}
test("retrieved official primary text supports narrow patent comparison",()=>{
 const result=groundedPatentComparison(task,plan,run());
 assert.equal(result?.success,true);
 assert.equal(result.source,"bounded-patent-comparison");
 assert.match(result.text,/раздел 3/);
 assert.match(result.text,/не условие о любом патентном споре/);
 assert.match(result.text,/отсутствие подразумеваемых патентных прав/);
 assert.match(result.text,/apache.org/);
 assert.match(result.text,/opensource.org/);
});
test("source-gated patent comparison fails closed on incomplete or unrelated evidence",()=>{
 assert.equal(groundedPatentComparison(task,plan,run(apache.replace("alleging that the Work or a Contribution incorporated within the Work","alleging that other material"),bsd)),null);
 assert.equal(groundedPatentComparison(task,plan,run(apache,bsd.replace("EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE","END"))),null);
 assert.equal(groundedPatentComparison("Сравни уведомления и патентные условия Apache 2.0 и BSD 3-Clause",plan,run()),null);
});
