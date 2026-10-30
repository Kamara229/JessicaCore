import test from "node:test";
import assert from "node:assert/strict";
import {fetchCanonicalMitText} from "../tools/canonicalMitText.js";
import {fetchCanonicalApacheText} from "../tools/canonicalApacheText.js";
const url="https://opensource.org/license/mit";
const parts=[
 "Permission is hereby granted, free of charge",
 "including without limitation the rights to use, copy, modify, merge, publish, distribute",
 "the above copyright notice and this permission notice shall be included",
 "the software is provided",
 "in no event shall the authors or copyright holders be liable",
 "example filler".repeat(80),
 "other dealings in the software"
];
const html="<main><h1>MIT License</h1><p>"+parts.join(" ")+"</p><footer>Navigation</footer></main>";
const stub=async()=>({ok:true,headers:{get:()=>"text/html"},text:async()=>html});
test("fixed-host MIT text is source checked and not navigation",async()=>{
 const result=await fetchCanonicalApacheText(url,stub);
 assert.equal(result?.success,true);
 assert.equal(result?.data?.url,url);
 assert.ok(result?.data?.content?.includes("in no event shall the authors"));
 assert.ok(!result?.data?.content?.includes("Navigation"));
});
test("wrong source or malformed data is rejected",async()=>{
 assert.equal(await fetchCanonicalMitText("https://example.org/other",stub),null);
 assert.equal(await fetchCanonicalMitText(url,async()=>({...await stub(),text:async()=>"short"})),null);
});
