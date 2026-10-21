import test from "node:test";
import assert from "node:assert/strict";
import {PROGRAMMING_LANGUAGES, findProgrammingLanguage} from "../core/foundations/programmingLanguageCatalog.js";
import {analyzeProgrammingTask,detectProgrammingLanguages} from "../core/foundations/programmingTaskContext.js";
import {buildProgrammingFoundation} from "../core/foundations/programmingPolicy.js";
import {buildPlannerInstructions} from "../core/planner/plannerPrompt.js";
import {buildTaskDecomposerInstructions} from "../core/decomposer/decomposerPrompt.js";
import {getComposerInstructions} from "../core/composer/composerPrompt.js";
import {buildAIResultValidatorMessages} from "../core/validator/aiResult/aiResultPrompt.js";
import {isCodeAttachmentTask,buildCodeAnalysisPrompt,analyzeCodeAttachments} from "../core/files/codeAnalysisFacade.js";
import {needsProgrammingRuntimeEvidence} from "../core/foundations/programmingLearningGate.js";

test("stable programming vocabulary spans common and niche languages", () => {
    assert.ok(PROGRAMMING_LANGUAGES.length >= 55);
    for(const name of ["python","javascript","typescript","java","kotlin","swift","go","rust",
        "c","cpp","csharp","php","ruby","sql","bash","powershell","solidity","cobol","fortran",
        "pascal","ocaml","zig","nim","haskell","graphql","xml","yaml","json"]){
        assert.ok(findProgrammingLanguage(name),name);
    }
});
test("recognize language and source file independently from UI locale", () => {
    assert.deepEqual(detectProgrammingLanguages("Напиши скрипт на Python").map(x=>x.id),["python"]);
    const names=detectProgrammingLanguages("Debug the code",[{name:"Example.kt"},{name:"source.rs"}]).map(x=>x.id);
    assert.ok(names.includes("kotlin"));
    assert.ok(names.includes("rust"));
    assert.ok(detectProgrammingLanguages("Напиши пример на C++").some(x=>x.id==="cpp"));
    assert.ok(detectProgrammingLanguages("Explain C# code").some(x=>x.id==="csharp"));
    const fenced="Please explain:\n" + String.fromCharCode(96).repeat(3) + "typescript\nconst x:number=1;\n" + String.fromCharCode(96).repeat(3);
    assert.ok(analyzeProgrammingTask(fenced).languages.includes("typescript"));
});
test("ordinary non-programming tasks do not receive accidental code policy", () => {
    assert.equal(analyzeProgrammingTask("Как тебя зовут, Джессика?").relevant,false);
    assert.equal(buildProgrammingFoundation("Помоги с рецептом ужина"),"");
    assert.equal(buildPlannerInstructions("Помоги с рецептом ужина").includes("JESSICA PROGRAMMING FOUNDATION"),false);
});
test("each decision stage receives immutable code literacy and safety policy", () => {
    const task="Напиши функцию на Rust и тесты";
    for(const msg of [buildTaskDecomposerInstructions(task),buildPlannerInstructions(task),
        getComposerInstructions(task),buildAIResultValidatorMessages(task,{}, {},{text:"fn add(a:i32,b:i32)->i32{a+b}"})[0].content]) {
        assert.match(msg,/JESSICA PROGRAMMING FOUNDATION/);
        assert.match(msg,/not proof|no interpreter|not a license/i);
    }
    assert.match(getComposerInstructions(task),/fenced code blocks/);
    assert.match(buildPlannerInstructions(task),/evidence.mode=none/);
});
test("plain uploaded source is code; ordinary financial CSV/JSON remains a document", () => {
    assert.equal(isCodeAttachmentTask("Объясни что делает функция",[{name:"module.py",text:"def f(): pass"}]),true);
    assert.equal(isCodeAttachmentTask("Debug this file",[{name:"core.kt",text:"fun f() = 1"}]),true);
    assert.equal(isCodeAttachmentTask("Подсчитай суммы",[{name:"budget.json",text:"{}"}]),false);
    assert.equal(isCodeAttachmentTask("Сравни документы",[{name:"letter.pdf",text:"some text"}]),false);
    assert.equal(isCodeAttachmentTask("Напиши тесты для SQL",[{name:"schema.sql",text:"SELECT 1"}]),true);
});
test("uploaded code is not executed or falsely verified", async () => {
    const source=[{name:"worker.py",text:"def f(x):\n  return x*2",truncated:false}];
    let captured;
    const result=await analyzeCodeAttachments({task:"Исправь функцию на Python",attachments:source},async(prompt,input)=>{
        captured={prompt,input};return "Corrected source:\n" + String.fromCharCode(96).repeat(3) + "python\ndef f(x): return x * 2\n" + String.fromCharCode(96).repeat(3);
    });
    assert.equal(result.success,true);
    assert.equal(result.mode,"code");
    assert.equal(result.validated,false);
    assert.equal(result.executed,false);
    assert.match(captured.input,/worker\.py/);
    assert.match(captured.prompt,/not executed|never execute|NOT executed|do not execute/i);
    assert.match(buildCodeAnalysisPrompt({task:"Review",attachments:source}),/SOURCE FILE/);
});
test("truncated source is labeled and empty source cannot create positive result", async () => {
    const result=await analyzeCodeAttachments({task:"Review Python",attachments:[{name:"code.py",text:"return x",truncated:true}]},
        async()=>"Potential undefined variable x");
    assert.equal(result.truncated,true);
    const missing=await analyzeCodeAttachments({task:"Review",attachments:[]},async()=> "unreachable");
    assert.equal(missing.success,false);
});

test("unexecuted code never turns into positive autonomous Experience", () => {
    assert.equal(needsProgrammingRuntimeEvidence("Напиши функцию на Python",
        {status:"COMPLETED",validation:{valid:true},toolResults:[]}),true);
    assert.equal(needsProgrammingRuntimeEvidence("Explain the principles of programming",
        {status:"COMPLETED",toolResults:[]}),false);
    assert.equal(needsProgrammingRuntimeEvidence("Напиши функцию на Python",
        {toolResults:[{tool:"run_tests",success:true,verified:true,validation:{valid:true}}]}),false);
    assert.equal(needsProgrammingRuntimeEvidence("Write JavaScript code",
        {toolResults:[{tool:"run_tests",success:true,verified:false,validation:{valid:true}}]}),true);
});

test("Russian legal code references are not programming tasks", () => {
    assert.equal(analyzeProgrammingTask("Объясни положения Гражданского кодекса РФ").relevant,false);
    assert.equal(needsProgrammingRuntimeEvidence("Гражданский кодекс РФ статья 309",{}),false);
});

test("code-edit followup keeps one semantic unit and avoids redundant LLM decomposition", async () => {
    const {tryNativeDecomposition}=await import("../core/decomposer/nativeDecomposer.js");
    const {contextualizeTask}=await import("../core/chat/conversationContext.js");
    const original="Теперь добавь обработку неправильного ввода и перепиши эту функцию на Kotlin.";
    const history=[
        {role:"user",content:"Напиши функцию факториала"},
        {role:"assistant",content:"def factorial(n):\\n    return 1"}
    ];
    const full=contextualizeTask(original,history);
    const result=tryNativeDecomposition({task:full});
    assert.equal(result.handled,true);
    assert.equal(result.source,"native-code-followup");
    assert.equal(result.decomposition.subtasks.length,1);
    assert.ok(result.decomposition.subtasks[0].text.includes("factorial"));
    assert.ok(result.decomposition.subtasks[0].text.endsWith(original));
});
