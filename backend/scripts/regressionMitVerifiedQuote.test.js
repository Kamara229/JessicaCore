import test from "node:test";
import assert from "node:assert/strict";
import {fetchCanonicalMitText} from "../tools/canonicalMitText.js";
import {fetchSpdxMitText,SPDX_MIT_RAW_URL,SPDX_MIT_REFERENCE_URL} from "../tools/canonicalMitSpdxText.js";
import {buildNativeSingleLicensePlan} from "../core/planner/nativeSingleLicensePlan.js";
import {extractMitCopyrightNotice} from "../core/composer/officialMitNoticeExcerpt.js";
import {validateSourceContent} from "../core/validator/sourceContentValidator.js";
import {localizeResponse} from "../core/language/responseLanguageService.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";

const task="Приведи дословную английскую цитату из лицензии MIT о сохранении copyright notice. Укажи официальный URL";
const OSI="https://opensource.org/license/mit";
const quote="The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.";
const original=[
 "MIT License",
 "Copyright (c) <year> <copyright holders>",
 'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:',
 quote,
 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'
].join("\n");
const fetched=async()=>({ok:true,headers:{get:()=> "text/plain; charset=utf-8"},text:async()=>original});

test("MIT known license routes to fixed OSI primary document",()=>{
 const plan=buildNativeSingleLicensePlan(task);
 assert.equal(plan.intent,"native_single_license_primary_text");
 assert.equal(plan.metadata.licenseId,"MIT");
 assert.equal(plan.steps[0].arguments.url,OSI);
});

test("failed OSI fetch falls back only to fully verified SPDX original",async()=>{
 const calls=[];
 const request=async(url)=>{
  calls.push(url);return url===OSI?{ok:false,status:403}:fetched();
 };
 const source=await fetchCanonicalMitText(OSI,request);
 assert.deepEqual(calls,[OSI,SPDX_MIT_RAW_URL]);
 assert.equal(source.success,true);
 assert.equal(source.data.url,SPDX_MIT_RAW_URL);
 assert.equal(source.data.referenceUrl,SPDX_MIT_REFERENCE_URL);
 const plan=buildNativeSingleLicensePlan(task);
 const run={results:[{tool:"web_fetch",...source}]};
 const excerpt=extractMitCopyrightNotice(task,plan,run);
 assert.equal(excerpt.source,"bounded-mit-notice-excerpt");
 assert.ok(excerpt.text.includes("“"+quote+"”"));
 assert.ok(excerpt.text.includes(SPDX_MIT_REFERENCE_URL));
 const verdict=await validateSourceContent(task,plan,run);
 assert.equal(verdict.valid,true,verdict.reason);
 const reply={success:true,mode:"single",validated:true,answerSource:excerpt.source,text:excerpt.text};
 let edited=0;
 const localized=await localizeResponse(reply,task,"ru",async()=>{edited++;return "Русский пересказ";});
 assert.equal(edited,0);
 assert.equal(localized.localizationStatus,"verified-original-quote");
 assert.equal(enforcePresentationLanguage(localized,{task,language:"ru"}).text,excerpt.text);
});

test("MIT source guards fail closed instead of triggering repeated replans",async()=>{
 const plan=buildNativeSingleLicensePlan(task);
 const broken=async()=>({ok:false,headers:{get:()=> "text/plain"},text:async()=>""});
 assert.equal(await fetchSpdxMitText(broken),null);
 assert.equal(await fetchCanonicalMitText(OSI,broken),null);
 const data=await fetchSpdxMitText(fetched);
 for(const corrupted of [
  {...data,data:{...data.data,url:"https://fake.example/mit"}},
  {...data,data:{...data.data,truncated:true}},
  {...data,data:{...data.data,content:original.replace(quote,"")}}
 ]){
  const run={results:[{tool:"web_fetch",...corrupted}]};
  assert.equal(extractMitCopyrightNotice(task,plan,run),null);
  const verdict=await validateSourceContent(task,plan,run);
  assert.equal(verdict.valid,false);
  assert.equal(verdict.shouldRetry,false);
 }
});

test("final validator accepts verified original without lossy AI claim snippets",async()=>{
 const {validateResult}=await import("../core/validator.js");
 const plan=buildNativeSingleLicensePlan(task);
 const source=await fetchSpdxMitText(fetched);
 const run={results:[{tool:"web_fetch",...source}]};
 const reply=extractMitCopyrightNotice(task,plan,run);
 assert.equal(reply.source,"bounded-mit-notice-excerpt");
 const verdict=await validateResult(task,plan,run,reply);
 assert.equal(verdict.valid,true,verdict.reason);
 assert.equal(verdict.validationMode,"bounded-primary-text");
 assert.equal(verdict.degraded,false);
 const corrupt={...reply,text:reply.text.replace("permission notice","permission text")};
 const rejected=await validateResult(task,plan,run,corrupt);
 assert.equal(rejected.valid,false);
 assert.equal(rejected.shouldRetry,false);
});
