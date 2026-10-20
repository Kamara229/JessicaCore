import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {tryNativeDecomposition} from "../core/decomposer/nativeDecomposer.js";
import {tryCreateKnownLicenseComparison} from "../core/planner/knownLicenseComparison.js";
import {verifyPairedSourceContent} from "../core/validator/pairedSourceContent.js";

const task = [
 "Сравни две лицензии программного обеспечения: Apache License 2.0 и BSD 3-Clause.",
 "Найди официальные тексты обеих лицензий и изучи их содержание.",
 "Сравни условия использования, распространения, сохранения уведомлений, патентные положения и ограничения.",
 "Представь результат в таблице, объясни главное различие и подтверди выводы официальными источниками.",
 "Если какие-либо условия не удалось проверить, прямо укажи это."
].join("\n");

test("full Apache/BSD user request stays one atomic comparison subtask", () => {
 const result=tryNativeDecomposition({task,memoryContext:{skills:[]}});
 assert.equal(result.handled,true);
 assert.equal(result.source,"native-known-license-cohesive");
 assert.equal(result.decomposition.subtasks.length,1);
 assert.equal(result.decomposition.subtasks[0].text,task);
 const plan=tryCreateKnownLicenseComparison({task:result.decomposition.subtasks[0].text});
 assert.equal(plan.handled,true);
 assert.deepEqual(plan.plan.steps.map(x=>x.tool),["web_fetch","web_fetch"]);
 assert.equal(plan.plan.intent,"native_known_license_comparison");
});

test("arbitrary unknown license names do not force the canonical route", () => {
 assert.equal(tryNativeDecomposition({
  task:"Сравни лицензии Foo License и Bar License",memoryContext:{skills:[]}
 }).handled,false);
});

test("explicit two user URLs retain their original higher priority", () => {
 const result=tryNativeDecomposition({task:"Сравни https://example.org/a и https://example.org/b"});
 assert.equal(result.source,"native-source-pair");
});

test("official domain alone does not validate incomplete license content", () => {
 const plan=tryCreateKnownLicenseComparison({task}).plan;
 const apache="Apache License Version 2.0. Grant of Patent License. Redistribution. You must retain, in the Source form of any Derivative Works, all copyright, patent, trademark, and attribution notices. If the Work includes a NOTICE text file, then any Derivative Works must include a readable copy of the attribution notices.";
 const bsd="Redistribution and use in source and binary forms with or without modification. Neither the name of the organization.";
 const result=(first,second)=>({
  results:plan.steps.map((step,index)=>({
   tool:"web_fetch",success:true,data:{
    url:step.arguments.url,
    content:(index===0?first:second).repeat(10),
    truncated:false
   }
  }))
 });
 assert.equal(verifyPairedSourceContent(plan,result("Apache License how to apply license",bsd)).valid,false);
 assert.equal(verifyPairedSourceContent(plan,result(apache,"BSD is permissive")).valid,false);
 assert.equal(verifyPairedSourceContent(plan,result(apache,bsd)).valid,true);
});
