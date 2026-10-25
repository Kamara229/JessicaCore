import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {extractOfficialLicenseExcerpt} from "../core/composer/officialLicenseExcerpt.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";

const task="Приведи дословно 8–12 английских слов из §3 Apache License 2.0 о прекращении патентных лицензий. Без перевода. Укажи официальный URL.";
const url="https://www.apache.org/licenses/LICENSE-2.0.txt";
const plan={
 intent:"native_single_license_primary_text",metadata:{licenseId:"Apache-2.0"},
 evidence:{mode:"source_content"},
 steps:[{id:"fetch_named_license",tool:"web_fetch",arguments:{url}}]
};
const textSource=[
 "Apache License Version 2.0, January 2004.",
 "3. Grant of Patent License. Subject to the terms and conditions of this License.",
 "If You institute patent litigation against any entity alleging that the Work",
 "constitutes direct or contributory patent infringement, then any patent",
 "licenses granted to You under this License for that Work shall terminate",
 "as of the date such litigation is filed.",
 "4. Redistribution. You may reproduce and distribute copies."
].join(" ");
const run=(body=textSource,options={})=>({
 success:true,results:[{
   tool:"web_fetch",success:true,
   data:{url,content:body,truncated:false,...options}
 }]
});

test("literal short quote is taken verbatim from original and verified without AI",async()=>{
 const answer=extractOfficialLicenseExcerpt(task,plan,run());
 assert.equal(answer?.source,"bounded-original-license-excerpt");
 const quote=answer.text.match(/“([^”]+)”/u)?.[1];
 assert.equal(quote?.split(/\s+/u).length,12);
 assert.ok(textSource.includes(quote));
 assert.match(answer.text,/https:\/\/www\.apache\.org\/licenses\/LICENSE-2\.0\.txt/u);
 const result=await composeAnswer(task,plan,run());
 assert.deepEqual(result,answer);
 const verified=await validateResult(task,plan,run(),answer);
 assert.equal(verified.valid,true,verified.reason);
 assert.equal(verified.validationMode,"bounded-primary-text");
});

test("wrong, incomplete or spoofed source fails closed",()=>{
 assert.equal(extractOfficialLicenseExcerpt(task,plan,run(textSource.replace(
  "shall terminate","may terminate"))),null);
 assert.equal(extractOfficialLicenseExcerpt(task,plan,run(textSource,{truncated:true})),null);
 assert.equal(extractOfficialLicenseExcerpt(task,plan,run(textSource,{url:"https://example.net/copy"})),null);
 assert.equal(extractOfficialLicenseExcerpt(task,plan,run(textSource.replace(
  "3. Grant of Patent License","3. Patent Summary"))),null);
 assert.equal(extractOfficialLicenseExcerpt("Сравни Apache и MIT",plan,run()),null);
});
test("manual translation cannot be validated as source excerpt",async()=>{
 const result=await validateResult(task,plan,run(),{
  success:true,source:"bounded-original-license-excerpt",
  text:"§3: лицензии прекращаются; "+url
 });
 assert.equal(result.valid,false);
});
