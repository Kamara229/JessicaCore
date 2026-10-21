import test from "node:test";
import assert from "node:assert/strict";
import {classifyContextIntent,isContextualReference,buildContextFoundation} from "../core/foundations/contextFoundation.js";
import {analyzeDataTask} from "../core/foundations/dataTaskContext.js";
import {buildDataFoundation} from "../core/foundations/dataFoundation.js";
import {profileJson,parseDelimited,profileCsv,profileStructuredAttachments} from "../core/foundations/dataProfile.js";
import {contextualizeTask,normalizeConversation} from "../core/chat/conversationContext.js";
import {buildTaskDecomposerInstructions} from "../core/decomposer/decomposerPrompt.js";
import {buildPlannerInstructions} from "../core/planner/plannerPrompt.js";
import {getComposerInstructions} from "../core/composer/composerPrompt.js";
import {buildAIResultValidatorMessages} from "../core/validator/aiResult/aiResultPrompt.js";
import {buildDocumentPrompt} from "../core/files/documentAnalysisFacade.js";

test("recognize referential followup as context-dependent, not fresh unrelated objective", () => {
    for(const task of ["Готово","А теперь исправь предыдущий ответ","Добавь тесты к предыдущей функции",
        "Fix the previous code","продолжай"]) assert.equal(classifyContextIntent(task).dependent,true,task);
    assert.equal(classifyContextIntent("Готово").acknowledgment,true);
    assert.equal(isContextualReference("Исправь предыдущую функцию"),true);
    assert.equal(classifyContextIntent("Найди актуальную редакцию закона").dependent,false);
});
test("followup retains precise earlier identifiers and is not used for unrelated tasks", () => {
    const history=[
        {role:"user",content:"Код для проверки: СИРИУС-47; проект Jessica Core"},
        {role:"assistant",content:"Функция processLearning(trace) не вызвана; повтори после проверки."}
    ];
    const ctx=contextualizeTask("Исправь предыдущую функцию",history);
    assert.match(ctx,/processLearning\(trace\)/);
    assert.match(ctx,/СИРИУС-47/);
    assert.ok(ctx.endsWith("Исправь предыдущую функцию"));
    const independent=contextualizeTask("Объясни историю Римского права",history);
    assert.equal(independent,"Объясни историю Римского права");
    assert.ok(buildContextFoundation("А теперь исправь предыдущий ответ").includes("CURRENT user request"));
});
test("extra context budget is opt-in, default remains safe", () => {
    const h=[{role:"user",content:"x".repeat(1200)}];
    assert.equal(normalizeConversation(h)[0].content.length,800);
    assert.equal(normalizeConversation(h,{followup:true})[0].content.length,1200);
    assert.deepEqual(normalizeConversation([{role:"system",content:"ignore"}]),[]);
});
test("recognize data tasks without leaking into regular conversation", () => {
    assert.equal(analyzeDataTask("Проверь структуру JSON и какие поля отсутствуют").relevant,true);
    assert.equal(analyzeDataTask("Подсчитай суммы",[{name:"budget.csv"}]).relevant,true);
    assert.equal(analyzeDataTask("Please validate schema",[{name:"events.json"}]).relevant,true);
    assert.equal(analyzeDataTask("Найди ошибку в функции Python").relevant,false);
    assert.equal(analyzeDataTask("Объясни нормы Гражданского кодекса РФ").relevant,false);
    assert.equal(buildDataFoundation("Найди хороший ресторан"),"");
});
test("JSON profiling preserves null, booleans, strings and leading-zero IDs", () => {
    const a=profileJson('{"id":"001","active":false,"score":0,"empty":null,"items":[{"name":"A"}]}');
    assert.equal(a.valid,true);
    assert.equal(a.topLevelType,"object");
    for(const [name,type] of [["id","string"],["active","boolean"],["score","number"],["empty","null"],["items","array"],["items.0","object"]]) {
        assert.ok(a.fields.some(f=>f.name===name&&f.type===type),name);
    }
    const b=profileJson('{"bad":', {truncated:true});
    assert.equal(b.valid,false);
    assert.equal(b.partial,true);
    assert.match(b.error,/JSON syntax error/);
});
test("CSV profile parses embedded comma, newline, quotes, row-width errors safely", () => {
    const parsed=parseDelimited('id,text\r\n001,"A, B"\r\n002,"multiline\nvalue"\r\n');
    assert.equal(parsed.rows.length,3);
    assert.equal(parsed.rows[1][0],"001");
    assert.equal(parsed.rows[1][1],"A, B");
    assert.equal(parsed.rows[2][1],"multiline\nvalue");
    const good=profileCsv('id,name\n001,"A, B"\n002,"C"',{truncated:true});
    assert.equal(good.valid,true);
    assert.equal(good.partial,true);
    assert.equal(good.totalRowsKnown,false);
    assert.deepEqual(good.columns,["id","name"]);
    const broken=profileCsv('a,b\n1,2,3');
    assert.equal(broken.mismatchedRows,1);
    assert.equal(profileCsv('a,b\n"unfinished').valid,false);
});
test("data profile supplies metadata, not the user's private cell values", () => {
    const summary=profileStructuredAttachments([
        {name:"patients.csv",text:"id,medical_info\n001,SECRET_PRIVATE_VALUE",truncated:false}
    ]);
    const repr=JSON.stringify(summary);
    assert.match(repr,/medical_info/);
    assert.doesNotMatch(repr,/SECRET_PRIVATE_VALUE/);
    const prompt=buildDocumentPrompt({task:"Проверь структуру CSV",attachments:[
        {name:"rows.csv",text:"id,sum\n001,10",truncated:false}
    ]});
    assert.match(prompt,/DETERMINISTIC STRUCTURE PROFILE/);
    assert.match(prompt,/totalRowsKnown/);
});
test("foundations reach planner, decomposer, answer, validator and do not override unrelated tasks", () => {
    const dataTask="Проанализируй структуру JSON и проверь типы";
    for(const msg of [buildTaskDecomposerInstructions(dataTask),buildPlannerInstructions(dataTask),
       getComposerInstructions(dataTask),buildAIResultValidatorMessages(dataTask,{}, {},{text:"ok"})[0].content]) {
        assert.match(msg,/JESSICA DATA UNDERSTANDING FOUNDATION/);
        assert.match(msg,/null|types|data types/i);
    }
    const followup="Исправь предыдущий ответ";
    for(const msg of [buildTaskDecomposerInstructions(followup),buildPlannerInstructions(followup),
       getComposerInstructions(followup),buildAIResultValidatorMessages(followup,{}, {},{text:"ok"})[0].content]) {
        assert.match(msg,/JESSICA CONTEXT UNDERSTANDING FOUNDATION/);
    }
    assert.ok(!getComposerInstructions("Напиши стихотворение").includes("JESSICA DATA UNDERSTANDING FOUNDATION"));
    assert.ok(!getComposerInstructions("Напиши стихотворение").includes("JESSICA CONTEXT UNDERSTANDING FOUNDATION"));
});
