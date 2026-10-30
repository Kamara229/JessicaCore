import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task="Сравни MIT и BSD 2-Clause по условиям распространения. Ответь только таблицей: права, уведомления, ограничения и официальные ссылки";
const plan=buildKnownLicenseComparisonPlan(task);
const mit=[
 "The MIT License Copyright YEAR HOLDER.",
 "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:",
 "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
 "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.",
 "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join(" ");
const bsd=[
 "The 2-Clause BSD License Copyright YEAR HOLDER.",
 "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:",
 "1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
 "2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.",
 "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.",
 "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
].join(" ");
const MIT="https://opensource.org/license/mit";
const BSD="https://opensource.org/license/bsd-2-clause";
function evidence(m=mit,b=bsd){
 return {results:plan.steps.map(step=>({tool:"web_fetch",success:true, data:{
  url:step.arguments.url,content:step.arguments.url===MIT?m:b,truncated:false
 }}))};
}

test("19:21 ordinary-chat MIT BSD-2 table-only incident passes source-bounded validation",async()=>{
 assert.equal(plan?.intent,"native_known_license_comparison");
 assert.deepEqual(new Set(plan.steps.map(x=>x.arguments.url)),new Set([MIT,BSD]));
 const answer=await composeAnswer(task,plan,evidence());
 assert.equal(answer?.source,"bounded-license-comparison");
 assert.ok(answer.text.startsWith("| Критерий | MIT License | BSD 2-Clause |"));
 assert.ok(!answer.text.includes("\n\n"));
 for(const row of ["Права","Уведомления","Ограничения","Официальные ссылки"])
  assert.ok(answer.text.includes("| "+row+" |"));
 assert.ok(answer.text.includes(MIT));
 assert.ok(answer.text.includes(BSD));
 const validation=await validateResult(task,plan,evidence(),answer);
 assert.equal(validation.valid,true,validation.reason);
 assert.equal(validation.validationMode,"bounded-primary-text");
});

test("MIT BSD2 refuses truncated, conflicting or unrelated source evidence",()=>{
 assert.equal(groundedPatentComparison(task,plan,evidence(mit.replace("sublicense, and/or sell copies","copy the software"),bsd)),null);
 assert.equal(groundedPatentComparison(task,plan,evidence(mit,bsd.replace("2. Redistributions in binary form","2. Redistribution without notice"))),null);
 const truncated=evidence();truncated.results[1].data.truncated=true;
 assert.equal(groundedPatentComparison(task,plan,truncated),null);
 const wrong=evidence();wrong.results[1].data.url="https://example.org/not-bsd";
 assert.equal(groundedPatentComparison(task,plan,wrong),null);
 assert.equal(groundedPatentComparison("Привет",plan,evidence()),null);
});
