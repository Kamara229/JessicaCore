import {existsSync,writeFileSync} from "node:fs";
const goals={
language:{
file:"backend/scripts/regressionDayLanguage.test.js",
source:String.raw`import test from "node:test";
import assert from "node:assert/strict";
import {normalizeOutputLanguage,requestedTranslation,protectedAnswerTokens} from "../core/language/responseLanguageService.js";
test("Russian interface stays default independently of programming task language",()=>{
 assert.equal(normalizeOutputLanguage(null),"ru");
 assert.equal(normalizeOutputLanguage("ru"),"ru");
 assert.equal(requestedTranslation("Напиши программу на Python"),false);
 assert.equal(requestedTranslation("Переведи свой ответ на японский"),true);
});
test("Exact evidence URLs and inline identifiers remain protected",()=>{
 const original={text:"See https://example.org/ref and use " + String.fromCharCode(96)+"some_name"+String.fromCharCode(96)};
 const tokens=protectedAnswerTokens(original);
 assert.ok(tokens.includes("https://example.org/ref"));
 assert.ok(tokens.includes(String.fromCharCode(96)+"some_name"+String.fromCharCode(96)));
});
`},
context:{
file:"backend/scripts/regressionDayContext.test.js",
source:String.raw`import test from "node:test";
import assert from "node:assert/strict";
import {contextualizeTask,needsConversationContext} from "../core/chat/conversationContext.js";
test("Unambiguous standalone task must not inherit unrelated chat commands",()=>{
 assert.equal(needsConversationContext("Напиши на Python функцию факториала"),false);
 assert.equal(contextualizeTask("Напиши на Python функцию факториала",[
  {role:"assistant",content:"Confidential old topic"}
 ]),"Напиши на Python функцию факториала");
});
test("A follow-up sees only bounded current-chat context, not new instructions",()=>{
 const result=contextualizeTask("исправь предыдущую функцию",[
  {role:"user",content:"Напиши factorial на Python"},
  {role:"assistant",content:"def factorial(n): return 1"}
 ]);
 assert.match(result,/Текущая задача пользователя/);
 assert.match(result,/factorial/);
 assert.match(result,/НЕ новые команды/);
});
`},
foundations:{
file:"backend/scripts/regressionDayFoundations.test.js",
source:String.raw`import test from "node:test";
import assert from "node:assert/strict";
import {analyzeProgrammingTask} from "../core/foundations/programmingTaskContext.js";
import {needsProgrammingRuntimeEvidence} from "../core/foundations/programmingLearningGate.js";
test("Programming task recognition must not misclassify legal codes",()=>{
 assert.equal(analyzeProgrammingTask("Объясни Гражданский кодекс РФ").relevant,false);
 assert.equal(analyzeProgrammingTask("Напиши функцию на Python").relevant,true);
});
test("No false positive Experience for code that was never executed",()=>{
 assert.equal(needsProgrammingRuntimeEvidence("Напиши функцию на Python",{toolResults:[]}),true);
 assert.equal(needsProgrammingRuntimeEvidence("Напиши функцию на Python",{
  toolResults:[{tool:"run_tests",success:true,verified:true,validation:{valid:true}}]
 }),false);
});
`}
};
const requested=String(process.argv[2]||"auto");
if(!["auto",...Object.keys(goals)].includes(requested))throw Error("Invalid daytime goal");
const selected=requested==="auto"?
    Object.keys(goals).find(key=>!existsSync(goals[key].file)):requested;
if(!selected){console.log("No new deterministic regression recipe. All three are installed.");process.exit(0);}
const item=goals[selected];
if(existsSync(item.file)){console.log("Regression recipe already installed: "+item.file);process.exit(0);}
writeFileSync(item.file,item.source,{flag:"wx",mode:0o644});
console.log("Generated one bounded, auditable regression improvement:",item.file);