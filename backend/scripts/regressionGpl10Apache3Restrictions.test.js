import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
import {gpl10PatentRestrictions} from "../core/composer/gpl10PatentRestrictions.js";
import {validateResult} from "../core/validator.js";

const task="Сравни патентные ограничения GPLv3 §10 и Apache 2.0 §3. Укажи различия и официальные URL";
const gpl=[
 "GNU General Public License Version 3. 8. Termination. 9. Acceptance Not Required for Having Copies.",
 "10. Automatic Licensing of Downstream Recipients.",
 "Each time you convey a covered work, the recipient automatically receives a license from the original licensors, to run, modify and propagate that work, subject to this License.",
 "You may not impose any further restrictions on the exercise of the rights granted or affirmed under this License.",
 "For example, you may not impose a license fee, royalty, or other charge for exercise of rights granted under this License, and you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the Program or any portion of it.",
 "11. Patents. Each contributor grants you a non-exclusive, worldwide, royalty-free patent license.",
 "END OF TERMS AND CONDITIONS"
].join("\n");
const apache=[
 "Apache License Version 2.0. 2. Grant of Copyright License.",
 "3. Grant of Patent License. Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license.",
 "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
 "4. Redistribution. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices. If the Work includes a NOTICE text file, then any Derivative Works must include a readable copy of the attribution notices.",
 "END OF TERMS AND CONDITIONS"
].join("\n");
const plan=buildKnownLicenseComparisonPlan(task);
const run=(g=gpl,a=apache)=>({results:plan.steps.map(step=>({
 tool:"web_fetch",success:true,data:{url:step.arguments.url,content:step.arguments.url.includes("gnu.org")?g:a,truncated:false}
}))});
test("GPL10 patent-claim restriction is compared with Apache3 litigation termination",async()=>{
 assert.equal(plan.intent,"native_known_license_comparison");
 const answer=groundedPatentComparison(task,plan,run());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.match(answer.text,/§10/u);
 assert.match(answer.text,/§3/u);
 assert.match(answer.text,/инициировать патентный иск/u);
 assert.match(answer.text,/прекращаются с даты подачи иска/u);
 assert.doesNotMatch(answer.text,/раздел 11, абзац 3 даёт участникам/u);
 assert.ok(answer.text.includes("https://www.gnu.org/licenses/gpl-3.0.txt"));
 assert.ok(answer.text.includes("https://www.apache.org/licenses/LICENSE-2.0.txt"));
 const verdict=await validateResult(task,plan,run(),answer);
 assert.equal(verdict.valid,true,verdict.reason);
 assert.equal(verdict.validationMode,"bounded-primary-text");
});
test("source content is necessary: removed GPL10 or Apache3 clause fails closed",()=>{
 assert.equal(gpl10PatentRestrictions(task,plan,run(gpl.replace("You may not impose any further restrictions","Restrictions may be imposed"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(gpl.replace("you may not initiate litigation","you may initiate litigation"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(gpl,apache.replace("shall terminate as of the date","shall remain until the date"))),null);
});
test("independently validated answer rejects altered patent effect",async()=>{
 const answer=groundedPatentComparison(task,plan,run());
 const changed={...answer,text:answer.text.replace("прекращаются с даты подачи иска","сохраняются после подачи иска")};
 const verdict=await validateResult(task,plan,run(),changed);
 assert.equal(verdict.valid,false);
});
test("general patent question stays out of GPL10-specific route",()=>{
 const generic="Сравни условия прекращения патентных лицензий в GPLv3 §11 и Apache 2.0 §3. Укажи официальные URL";
 assert.equal(gpl10PatentRestrictions(generic,plan,run()),null);
});
