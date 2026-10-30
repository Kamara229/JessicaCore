import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {buildNativeSingleLicensePlan} from "../core/planner/nativeSingleLicensePlan.js";
import {extractGplSection4Excerpt} from "../core/composer/officialGplSection4Excerpt.js";
import {validateSourceContent} from "../core/validator/sourceContentValidator.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";

const URL="https://www.gnu.org/licenses/gpl-3.0.txt";
const copyright="conspicuously and appropriately publish on each copy an appropriate copyright notice";
const text=[
 "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.",
 "2. Basic Permissions. Some introductory details about distribution.",
 "3. Protecting Users Legal Rights From Anti-Circumvention Law.",
 "4. Conveying Verbatim Copies.",
 "You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you "+copyright+";",
 "keep intact all notices stating that this License and any non-permissive terms added in accord with section 7 apply to the code;",
 "keep intact all notices of the absence of any warranty; and give all recipients a copy of this License along with the Program.",
 "You may charge any price or no price for each copy that you convey.",
 "5. Conveying Modified Source Versions.",
 "The work must carry prominent notices stating that you modified it.",
 "END OF TERMS AND CONDITIONS"
].join(" ");
const task="Приведи дословную цитату из §4 GPL 3.0 о copyright notice. Укажи официальный URL";
const taskV3="Приведи дословную цитату из §4 GPLv3 о сохранении уведомлений об авторском праве. Укажи официальный URL. Не пересказывай";
function run(content=text,url=URL,truncated=false){return {results:[{tool:"web_fetch",success:true,data:{url,content,truncated}}]};}

test("both precise tasks select one official GPLv3 source without web search",()=>{
 for(const q of [task,taskV3]){
  const planned=tryCreateNativeSkillPlan({task:q,executionExperience:{skills:[]}});
  assert.equal(planned.handled,true,planned.validation?.text);
  assert.equal(planned.source,"native-single-license-source");
  assert.equal(planned.plan.steps.length,1);
  assert.equal(planned.plan.steps[0].tool,"web_fetch");
  assert.equal(planned.plan.steps[0].arguments.url,URL);
 }
});

test("original section 4 quote is exact, sourced and independently validated",async()=>{
 const plan=buildNativeSingleLicensePlan(task);
 const excerpt=extractGplSection4Excerpt(task,plan,run());
 assert.equal(excerpt?.source,"bounded-gpl-section4-excerpt");
 assert.equal(excerpt.text,"§4 GPLv3: “"+copyright+"”.\nОфициальный URL: "+URL);
 const source=await validateSourceContent(task,plan,run());
 assert.equal(source.valid,true,source.reason);
 const answer=await composeAnswer(task,plan,run());
 assert.deepEqual(answer,excerpt);
 const validated=await validateResult(task,plan,run(),answer);
 assert.equal(validated.valid,true,validated.reason);
 assert.equal(validated.validationMode,"bounded-primary-text");
});

test("source-gated quote never accepts changed section, truncated page, or wrong URL",async()=>{
 const plan=buildNativeSingleLicensePlan(task);
 assert.equal(extractGplSection4Excerpt(task,plan,run(text.replace(copyright,"no copyright notice"))),null);
 assert.equal(extractGplSection4Excerpt(task,plan,run(text.replace("4. Conveying Verbatim Copies.","4. Other Content."))),null);
 assert.equal(extractGplSection4Excerpt(task,plan,run(text,URL,true)),null);
 assert.equal(extractGplSection4Excerpt(task,plan,run(text,"https://example.org/gpl")),null);
 const invalid=await validateSourceContent(task,plan,run(text.replace(copyright,"no copyright notice")));
 assert.equal(invalid.valid,false);
 assert.equal(invalid.shouldRetry,false);
 assert.equal(extractGplSection4Excerpt("Найди историю GPL 3",plan,run()),null);
});
