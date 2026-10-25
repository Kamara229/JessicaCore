import test from "node:test";
import assert from "node:assert/strict";
import {checkOriginalEnglishQuote} from "../core/validator/originalEnglishQuoteGuard.js";
import {validateResult} from "../core/validator.js";
const task="Найти в §3 Apache License 2.0 точную формулировку о прекращении патентных лицензий и скопировать 8–12 английских слов.";
const result={
 success:true,
 results:[{
   tool:"web_fetch",success:true,
   data:{url:"https://www.apache.org/licenses/LICENSE-2.0.txt",
     content:"Apache License Version 2.0, January 2004. Only intro available.",
     truncated:false}
 }]
};
test("decomposed exact-wording request rejects unrelated source snippets",async()=>{
 const bad={success:true,text:"Источник загружен.\nVersion 2.0, January 2004\nhttp://www.apache.org"};
 assert.equal(checkOriginalEnglishQuote(task,result,bad.text)?.valid,false);
 const check=await validateResult(task,{evidence:{mode:"source_content"}},result,bad);
 assert.equal(check.valid,false);
 assert.equal(check.failureScope,"presentation");
 assert.equal(check.validationMode,"original-english-quote");
});
test("original-language requirement survives rephrasing",()=>{
 assert.equal(checkOriginalEnglishQuote(task,result,"Лицензия раздел 3")?.valid,false);
 assert.equal(checkOriginalEnglishQuote("Перескажи по-русски условия раздела 3",result,"Пересказ"),null);
});
