import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";

const task = "Сравни MIT и BSD 3-Clause. Ответь только таблицей";
const plan = buildKnownLicenseComparisonPlan(task);
const mit = [
 "The MIT License Copyright YEAR PERSON.",
 "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:",
 "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
 "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.",
 "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join(" ");
const bsd = [
 "The 3-Clause BSD License. Copyright YEAR PERSON.",
 "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:",
 "1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
 "2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.",
 "3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.",
 "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.",
 "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
].join(" ");
function evidence(first=mit,second=bsd) {
 return {results:plan.steps.map((step,i) => ({
   tool:"web_fetch",success:true, data:{
     url:step.arguments.url,content:i===0?first:second,truncated:false
   }
 }))};
}

test("exact failed ordinary-chat request produces only verified table",async()=>{
 assert.equal(plan?.intent,"native_known_license_comparison");
 assert.deepEqual(plan.steps.map(step=>step.arguments.url),[
  "https://opensource.org/license/mit",
  "https://opensource.org/license/bsd-3-clause"
 ]);
 const answer=await composeAnswer(task,plan,evidence());
 assert.equal(answer?.source,"bounded-license-comparison");
 assert.equal(answer?.success,true);
 assert.ok(answer.text.startsWith("| Критерий | MIT License | BSD 3-Clause |"));
 assert.ok(answer.text.endsWith("|"));
 assert.ok(!answer.text.includes("\n\n"));
 for(const item of ["Разрешённые действия","Уведомления при распространении","Использование имён авторов","Гарантии","Официальные тексты"])
  assert.ok(answer.text.includes("| "+item+" |"));
 assert.ok(answer.text.includes("https://opensource.org/license/mit"));
 assert.ok(answer.text.includes("https://opensource.org/license/bsd-3-clause"));
 const validated=await validateResult(task,plan,evidence(),answer);
 assert.equal(validated.valid,true,validated.reason);
 assert.equal(validated.validationMode,"bounded-primary-text");
});

test("no legal comparison without both complete unchanged source texts",()=>{
 assert.equal(groundedPatentComparison(task,plan,evidence(mit.replace("sublicense, and/or sell copies","make copies"),bsd)),null);
 assert.equal(groundedPatentComparison(task,plan,evidence(mit,bsd.replace("3. Neither the name","3. Anyone may use the name"))),null);
 const incomplete=evidence();incomplete.results[1].data.truncated=true;
 assert.equal(groundedPatentComparison(task,plan,incomplete),null);
 const wrong=evidence();wrong.results[1].data.url="https://untrusted.example/license/bsd-3-clause";
 assert.equal(groundedPatentComparison(task,plan,wrong),null);
 assert.equal(groundedPatentComparison("Привет!",plan,evidence()),null);
});
