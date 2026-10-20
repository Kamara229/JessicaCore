import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {verifyPairedSourceContent} from "../core/validator/pairedSourceContent.js";
import {correctNoticeFileClaim} from "../core/composer/comparisonConditionalNoticeMeaning.js";

const task="Сравни лицензии Apache License 2.0 и BSD 3-Clause: использование, распространение, уведомления.";
const plan=buildKnownLicenseComparisonPlan(task);
const apache=[
 "Apache License Version 2.0. 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license.",
 "3. Grant of Patent License. 4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in Source or Object form.",
 "You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices.",
 "If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file."
].join(" ");
const bsd=[
 "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met.",
 "Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
 "Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer.",
 "Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products without specific prior written permission."
].join(" ");
const run=(text)=>({results:plan.steps.map((step,index)=>({
 tool:"web_fetch",success:true,data:{url:step.arguments.url,
  content:index===0?text:bsd,truncated:false}
}))});

test("canonical Apache source is the complete official plain-text document",()=>{
 assert.equal(plan.steps[0].arguments.url,"https://www.apache.org/licenses/LICENSE-2.0.txt");
 assert.equal(verifyPairedSourceContent(plan,run(apache)).valid,true);
});
test("reject scraped license text that drops numbered NOTICE clauses",()=>{
 const truncated=apache.replace(/You must retain,[^.]+\./u,"")
   .replace(/If the Work includes a NOTICE[^.]+\./u,"")
   +(" Background definitions and general information about a legal agreement. ".repeat(8));
 const checked=verifyPairedSourceContent(plan,run(truncated));
 assert.equal(checked.valid,false);
 assert.match(checked.reason,/уведомлен/iu);
});
test("correct full NOTICE-file preservation overclaim only when source proves actual duty",()=>{
 const overclaim="При распространении требуется сохранять оригинальный файл **NOTICE** и включать его текст в копии.";
 const result=correctNoticeFileClaim(overclaim,apache);
 assert.match(result,/читаемую копию относящихся к ней атрибуционных уведомлений/iu);
 assert.doesNotMatch(result,/сохранять оригинальный файл/iu);
});
