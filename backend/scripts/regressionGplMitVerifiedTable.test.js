import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {buildKnownLicenseComparisonPlan} from "../core/planner/knownLicenseComparison.js";
import {groundedPatentComparison} from "../core/composer/nativePatentComparison.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";

const task="Сравни GPLv3 и MIT по патентным правам. Ответь только таблицей: предоставление прав, прекращение, исключения и официальные ссылки. Не придумывай условия";
const G="https://www.gnu.org/licenses/gpl-3.0.txt";
const M="https://opensource.org/license/mit";
const gpl=[
 "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.",
 "8. Termination. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11). However, if you cease all violation of this License, then your license is reinstated.",
 "10. Automatic Licensing of Downstream Recipients. You may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the Program.",
 "11. Patents. Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims. END OF TERMS AND CONDITIONS"
].join(" ");
const mit=[
 "MIT License Copyright YEAR HOLDER.",
 "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:",
 "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.",
 "THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.",
 "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
].join(" ");
const plan=buildKnownLicenseComparisonPlan(task);
function run(g=gpl,m=mit){return {results:plan.steps.map(s=>({
 tool:"web_fetch",success:true,data:{url:s.arguments.url,
 content:s.arguments.url===G?g:m,truncated:false}
}))};}

test("ordinary chat failed GPLv3 MIT task returns fully checked table",async()=>{
 assert.equal(plan?.intent,"native_known_license_comparison");
 assert.deepEqual(new Set(plan.steps.map(s=>s.arguments.url)),new Set([G,M]));
 const answer=await composeAnswer(task,plan,run());
 assert.equal(answer?.source,"bounded-patent-comparison");
 assert.ok(answer.text.startsWith("| Критерий | GPLv3 | MIT License |"));
 assert.ok(!answer.text.includes("\n\n"));
 for(const title of ["Предоставление патентных прав","Прекращение","Патентные иски и исключения","Официальные ссылки"])
    assert.ok(answer.text.includes("| "+title+" |"));
 assert.ok(answer.text.includes("§11"));
 assert.ok(answer.text.includes("§8"));
 assert.ok(answer.text.includes("§10"));
 assert.ok(answer.text.includes(G)); assert.ok(answer.text.includes(M));
 const verdict=await validateResult(task,plan,run(),answer);
 assert.equal(verdict.valid,true,verdict.reason);
 assert.equal(verdict.validationMode,"bounded-primary-text");
});

test("source validation refuses incomplete, altered and unrelated license text",()=>{
 assert.equal(groundedPatentComparison(task,plan,run(gpl.replace("including a cross-claim","excluding a cross-claim"),mit)),null);
 assert.equal(groundedPatentComparison(task,plan,run(gpl,mit.replace("sublicense, and/or sell copies","copy files"))),null);
 assert.equal(groundedPatentComparison(task,plan,run(gpl,mit.replace("THE SOFTWARE IS PROVIDED AS IS","A patent license is hereby granted. THE SOFTWARE IS PROVIDED AS IS"))),null);
 const truncated=run(); truncated.results[0].data.truncated=true;
 assert.equal(groundedPatentComparison(task,plan,truncated),null);
 const wrong=run(); wrong.results[0].data.url="https://not.gnu.example/gpl-3.0.txt";
 assert.equal(groundedPatentComparison(task,plan,wrong),null);
 assert.equal(groundedPatentComparison("Привет",plan,run()),null);
});
