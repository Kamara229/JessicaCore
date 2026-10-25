import test from "node:test";
import assert from "node:assert/strict";
import {checkOriginalEnglishQuote} from "../core/validator/originalEnglishQuoteGuard.js";
import {validateResult} from "../core/validator.js";
const task="Приведи дословную английскую фразу из оригинала и ссылку.";
const sentence="Relevant patent permissions end from the date a qualifying court action is filed";
const run={success:true,results:[{tool:"web_fetch",success:true,
  data:{url:"https://example.org/license",content:"Section three. "+sentence+"."}}]};
test("Russian translation cannot pass a verbatim English source request",async()=>{
 const answer="Патентные разрешения прекращаются с даты подачи иска. URL: https://example.org/license";
 assert.equal(checkOriginalEnglishQuote(task,run,answer).valid,false);
 const result=await validateResult(task,{evidence:{mode:"source_content"}},run,{success:true,text:answer});
 assert.equal(result.valid,false);
 assert.equal(result.failureScope,"presentation");
});
test("English quote must come from fetched source text",()=>{
 assert.equal(checkOriginalEnglishQuote(task,run,sentence),null);
 assert.equal(checkOriginalEnglishQuote(task,run,"Relevant patent permissions end from the date some different claim is filed").valid,false);
 assert.equal(checkOriginalEnglishQuote(task,{results:[]},sentence).valid,false);
});
test("Normal explanations remain unaffected",()=>{
 assert.equal(checkOriginalEnglishQuote("Объясни смысл условия по-русски",run,"Значение объясняется."),null);
});
