import test from "node:test";
import assert from "node:assert/strict";
import {fetchCanonicalApacheText} from "../tools/canonicalApacheText.js";
import {extractOfficialLicenseExcerpt} from "../core/composer/officialLicenseExcerpt.js";

const URL="https://www.apache.org/licenses/LICENSE-2.0.txt";
const textSource=[
 "Apache License Version 2.0 January 2004.",
 "1. Definitions. "+("Ordinary permission text. ".repeat(260)),
 "3. Grant of Patent License. Subject to the terms and conditions of this License.",
 "If You institute patent litigation against any entity alleging that the Work",
 "constitutes direct or contributory patent infringement, then any patent",
 "licenses granted to You under this License for that Work shall terminate",
 "as of the date such litigation is filed.",
 "4. Redistribution. "+("Terms of redistribution. ".repeat(55))
].join("\n");
const task="Найти в §3 Apache License 2.0 точную формулировку, где упоминается прекращение патентных лицензий, и скопировать 8–12 английских слов из этого места. Указать официальный URL лицензии.";
const plan={
 intent:"native_single_license_primary_text", metadata:{licenseId:"Apache-2.0"},
 steps:[{id:"fetch_named_license",tool:"web_fetch",arguments:{url:URL}}]
};

test("canonical text returns full original and literal excerpt under decomposed wording",async()=>{
 assert.ok(textSource.length>6000);
 let calls=0;
 const reader=async (url,options)=>{
   calls++; assert.equal(url,URL);
   assert.equal(options.redirect,"error");
   return {ok:true,headers:{get:()=>"text/plain; charset=utf-8"},
       text:async()=>textSource};
 };
 const verified=await fetchCanonicalApacheText(URL,reader);
 assert.equal(calls,1);
 assert.equal(verified?.data?.content,textSource);
 const run={success:true,results:[{tool:"web_fetch",...verified}]};
 const excerpt=extractOfficialLicenseExcerpt(task,plan,run);
 assert.equal(excerpt?.source,"bounded-original-license-excerpt");
 assert.match(excerpt.text,/licenses granted to You under this License for that Work shall terminate/u);
 assert.equal(excerpt.text.match(/“([^”]+)”/u)[1].split(" ").length,12);
});
test("unknown or invalid plain text fails closed",async()=>{
 const valid=()=>({ok:true,headers:{get:()=>"text/plain"},text:async()=>textSource});
 assert.equal(await fetchCanonicalApacheText("https://example.org/a",valid),null);
 assert.equal(await fetchCanonicalApacheText(URL,async()=>({
   ok:true,headers:{get:()=>"text/html"},text:async()=>textSource
 })),null);
 assert.equal(await fetchCanonicalApacheText(URL,async()=>({
   ok:true,headers:{get:()=>"text/plain"},
   text:async()=>textSource.replace("shall terminate","might terminate")
 })),null);
});
test("registered web_fetch prefers canonical text instead of third party renderer",async()=>{
 const original=globalThis.fetch;
 let count=0;
 globalThis.fetch=async (url)=>{
   count++; assert.equal(url,URL);
   return {ok:true,headers:{get:()=>"text/plain"},text:async()=>textSource};
 };
 try {
   await import("../tools/webFetchTool.js");
   const {executeTool}=await import("../tools/toolRegistry.js");
   const result=await executeTool("web_fetch",{url:URL});
   assert.equal(result.success,true);
   assert.equal(result.data.content,textSource);
   assert.equal(count,1);
 }finally{globalThis.fetch=original;}
});
