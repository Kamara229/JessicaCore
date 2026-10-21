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
