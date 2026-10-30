import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
import {compareLicenseRestoration} from "../core/composer/licenseRestoration.js";
import {validateResult} from "../core/validator.js";

const task="Сравни восстановление прав по GPLv3 §8 и прекращение патентной лицензии по Apache 2.0 §3. Укажи официальные URL";
const gpl="GNU General Public License version 3.\n8. Termination. You may not propagate or modify a covered work except as expressly provided under this License. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11). However, if you cease all violation of this License, then your license from a particular copyright holder is reinstated (a) provisionally, unless and until the copyright holder explicitly and finally terminates your license, and (b) permanently, if the copyright holder fails to notify you of the violation by some reasonable means prior to 60 days after the cessation. Moreover, your license from a particular copyright holder is reinstated permanently if the copyright holder notifies you of the violation by some reasonable means, this is the first time you have received notice of violation of this License (for any work) from that copyright holder, and you cure the violation prior to 30 days after your receipt of the notice.\n9. Acceptance Not Required for Having Copies.";
const apache="Apache License Version 2.0. 3. Grant of Patent License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive patent license. If You institute patent litigation against any entity including a cross-claim or counterclaim in a lawsuit alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed. 4. Redistribution. You must retain in the source form of derivative works all copyright, patent, trademark, and attribution notices. If the Work includes a NOTICE text file, then any Derivative Works must include a readable copy of the attribution notices.";
const plan=buildKnownLicenseComparisonPlan(task);
const run= (textG=gpl,textA=apache)=>({results:plan.steps.map(step=>({
 tool:"web_fetch",success:true,data:{url:step.arguments.url,
  content:/gnu.org/.test(step.arguments.url)?textG:textA,truncated:false}
}))});

test("real restoration request has its own bounded two-source explanation",async()=>{
 assert.equal(plan.intent,"native_known_license_comparison");
 const proof=groundedPatentComparison(task,plan,run());
 assert.equal(proof?.source,"bounded-patent-comparison");
 assert.match(proof.text,/30 дней/u);
 assert.match(proof.text,/60 дней/u);
 assert.match(proof.text,/предварительно/u);
 assert.ok(proof.text.includes("https://www.gnu.org/licenses/gpl-3.0.txt"));
 assert.ok(proof.text.includes("https://www.apache.org/licenses/LICENSE-2.0.txt"));
 assert.doesNotMatch(proof.text,/Раздел 10 запрещает/u);
 const verified=await validateResult(task,plan,run(),proof);
 assert.equal(verified.valid,true,verified.reason);
 assert.equal(verified.validationMode,"bounded-primary-text");
});
test("missing restoration provisions do not fall back to generic patent overview",()=>{
 const noRestoration=gpl.replace("prior to 60 days after the cessation","notice in time");
 assert.equal(groundedPatentComparison(task,plan,run(noRestoration)),null);
 const noApacheTermination=apache.replace("shall terminate as of the date such litigation is filed","may continue");
 assert.equal(compareLicenseRestoration(task,plan,run(gpl,noApacheTermination)),null);
 const other="Сравни условия прекращения патентных лицензий в GPLv3 §11 и Apache 2.0 §3. Укажи официальные URL";
 assert.equal(compareLicenseRestoration(other,plan,run()),null);
});
test("modified bounded restoration answer fails verification",async()=>{
 const verified=groundedPatentComparison(task,plan,run());
 const incorrect={...verified,text:verified.text.replace("30 дней","40 дней")};
 const result=await validateResult(task,plan,run(),incorrect);
 assert.equal(result.valid,false);
});
