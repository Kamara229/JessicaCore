import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {buildNativeSingleLicensePlan} from "../core/planner/nativeSingleLicensePlan.js";
const task="Приведи дословно 8–12 английских слов подряд из §3 Apache License 2.0 о прекращении патентных лицензий. Без перевода. Укажи официальный URL.";
test("known Apache license text is fetched directly without source selector",()=>{
  const result=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
  assert.equal(result.handled,true,result.validation?.text);
  assert.equal(result.source,"native-single-license-source");
  assert.equal(result.plan.steps.length,1);
  assert.equal(result.plan.steps[0].tool,"web_fetch");
  assert.equal(result.plan.steps[0].arguments.url,"https://www.apache.org/licenses/LICENSE-2.0.txt");
  assert.equal(result.plan.evidence.mode,"source_content");
});
test("ambiguous or unspecified references are not treated as primary sources",()=>{
  assert.equal(buildNativeSingleLicensePlan("Сравни Apache 2.0 и GPLv3 по патентам"),null);
  assert.equal(buildNativeSingleLicensePlan("Что такое Apache License 2.0?"),null);
  assert.equal(buildNativeSingleLicensePlan("Сравни два неизвестных документа"),null);
});
