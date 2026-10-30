import test from "node:test";
import assert from "node:assert/strict";
import {tryNativeDecomposition} from "../core/decomposer/nativeDecomposer.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {fetchCanonicalGpl3Text} from "../tools/canonicalGpl3Text.js";
import {extractGplSection4Excerpt} from "../core/composer/officialGplSection4Excerpt.js";
import {validateSourceContent} from "../core/validator/sourceContentValidator.js";

const task="Приведи дословную цитату из §4 GPL 3.0 о copyright notice. Укажи официальный URL";
const alternate="Приведи дословную цитату из §4 GPLv3 о сохранении уведомлений об авторском праве. Укажи официальный URL. Не пересказывай";
const URL="https://www.gnu.org/licenses/gpl-3.0.txt";
const content=[
 "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.",
 "2. Basic Permissions.",
 "4. Conveying Verbatim Copies.",
 "You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice;",
 "keep intact all notices stating that this License and any non-permissive terms added in accord with section 7 apply to the code;",
 "keep intact all notices of the absence of any warranty; and give all recipients a copy of this License along with the Program.",
 "You may charge any price or no price for each copy that you convey.",
 "5. Conveying Modified Source Versions.",
 "6. Conveying Non-Source Forms.",
 "11. Patents.",
 "End of Terms and Conditions."
].join("\n")+"\n"+(" Additional official-text filler. ".repeat(800));
const fetchMock=async()=>({ok:true,headers:{get:()=> "text/plain; charset=utf-8"},text:async()=>content});

test("original single-license citation survives decomposition without translation",()=>{
 for(const prompt of [task,alternate]){
  const decomp=tryNativeDecomposition({task:prompt});
  assert.equal(decomp.handled,true);
  assert.equal(decomp.source,"native-single-license-cohesive");
  assert.equal(decomp.decomposition.subtasks.length,1);
  assert.equal(decomp.decomposition.subtasks[0].text,prompt);
  const plan=tryCreateNativeSkillPlan({task:prompt,executionExperience:{skills:[]}});
  assert.equal(plan.handled,true,plan.validation?.text);
  assert.equal(plan.plan.steps[0].arguments.url,URL);
 }
});

test("fixed URL GPL download validates entire original before bounded quote",async()=>{
 const fetched=await fetchCanonicalGpl3Text(URL,fetchMock);
 assert.equal(fetched?.success,true);
 assert.equal(fetched.data.url,URL);
 assert.equal(fetched.data.truncated,false);
 const plan=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}}).plan;
 const run={results:[{tool:"web_fetch",...fetched}]};
 const quote=extractGplSection4Excerpt(task,plan,run);
 assert.equal(quote?.source,"bounded-gpl-section4-excerpt");
 assert.ok(quote.text.includes("conspicuously and appropriately publish on each copy an appropriate copyright notice"));
 const verdict=await validateSourceContent(task,plan,run);
 assert.equal(verdict.valid,true,verdict.reason);
});

test("canonical fetch rejects redirected, broken, or modified license content",async()=>{
 assert.equal(await fetchCanonicalGpl3Text("https://example.org/gpl",fetchMock),null);
 assert.equal(await fetchCanonicalGpl3Text(URL,async()=>({ok:false})),null);
 assert.equal(await fetchCanonicalGpl3Text(URL,async()=>({ok:true,headers:{get:()=> "text/html"},text:async()=>content})),null);
 assert.equal(await fetchCanonicalGpl3Text(URL,async()=>({ok:true,headers:{get:()=> "text/plain"},text:async()=>content.replace("conspicuously and appropriately publish on each copy an appropriate copyright notice","") })),null);
});
