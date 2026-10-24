import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildMitPatentQuestionPlan,tryCreateKnownLicenseComparison} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
import {validateResult} from "../core/validator.js";

const task="Доказывает ли отсутствие прямого патентного гранта MIT отсутствие патентных прав? Обоснуй по тексту MIT.";
const official="https://opensource.org/license/mit";
const source=[
"The MIT License (MIT). Copyright (c) 2026 Example.",
"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:",
"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
"THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.",
"IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join("\n");
const plan=buildMitPatentQuestionPlan(task);
const run=(content=source)=>({success:true,results:[{tool:"web_fetch",success:true,
 data:{url:official,content,truncated:false}}]});

test("MIT alone reads MIT License text rather than unrelated MIT Institute patents",()=>{
 assert.equal(plan?.intent,"native_mit_patent_question");
 assert.deepEqual(plan.steps.map(x=>x.arguments.url),[official]);
 assert.equal(tryCreateKnownLicenseComparison({task}).handled,true);
 assert.equal(buildMitPatentQuestionPlan("Объясни MIT Institute patent technology transfer"),null);
 assert.equal(buildMitPatentQuestionPlan("Сравни Apache 2.0 и MIT License патентные условия"),null);
});
test("MIT answer provides bounded reasoning and correct source",async()=>{
 const answer=groundedPatentComparison(task,plan,run());
 assert.equal(answer?.source,"bounded-mit-patent-question");
 assert.match(answer.text,/Нет\./);
 assert.match(answer.text,/не доказывает отсутствие патентных прав/);
 assert.match(answer.text,/подразумеваемая патентная лицензия/);
 assert.match(answer.text,/https:\/\/opensource.org\/license\/mit/);
 assert.ok(answer.text.length>450);
 const validation=await validateResult(task,plan,run(),answer);
 assert.equal(validation.valid,true);
 assert.equal(validation.validationMode,"bounded-primary-text");
});
test("wrong URL, incomplete grant or forged patent clause never pass verification",()=>{
 assert.equal(groundedPatentComparison(task,plan,run("Permission is hereby granted")),null);
 assert.equal(groundedPatentComparison(task,plan,run(source.replace("OTHER DEALINGS IN THE SOFTWARE","OTHER ACTIONS"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(source.replace("THE SOFTWARE IS PROVIDED AS IS","A patent license is granted. THE SOFTWARE IS PROVIDED AS IS"))),null);
 assert.equal(groundedPatentComparison(task,plan,{success:true,results:[{tool:"web_fetch",success:true,data:{url:"https://web.mit.edu",content:source}}]}),null);
});
