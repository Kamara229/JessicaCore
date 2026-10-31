import test from "node:test";
import assert from "node:assert/strict";
import {parseSafeAccountTransferTask} from "../core/native/safeAccountTransferIntent.js";
import {decomposeTask} from "../core/taskDecomposer.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {safeAccountTransferSqlTool} from "../tools/safeAccountTransferSqlTool.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
import {localizeResponse} from "../core/language/responseLanguageService.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";

const task="Исправь PL/pgSQL: блокируй оба счёта по ID, проверь сумму > 0, их наличие и остаток. Без COMMIT внутри.";

test("safe-account-transfer PL/pgSQL uses native decomposition and planner",async()=>{
 assert.deepEqual(parseSafeAccountTransferTask(task),{});
 const decomposed=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(decomposed.native,true);
 assert.equal(decomposed.decomposition.subtasks[0].text,task);
 const routed=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(routed.handled,true);
 assert.equal(routed.plan.intent,"native_safe_account_transfer_sql");
 const step=routed.plan.steps[0];
 assert.equal(step.tool,"safe_account_transfer_sql");
 const output=safeAccountTransferSqlTool(step.arguments);
 const sql=output.data.sql;
 assert.match(sql,/LEAST\(p_from_account, p_to_account\).*FOR UPDATE/);
 assert.match(sql,/GREATEST\(p_from_account, p_to_account\).*FOR UPDATE/);
 assert.match(sql,/p_amount IS NULL OR p_amount <= 0/);
 assert.match(sql,/IF NOT FOUND THEN/);
 assert.match(sql,/IF v_balance < p_amount/);
 assert.match(sql,/UPDATE accounts SET balance = balance - p_amount/);
 assert.match(sql,/UPDATE accounts SET balance = balance \+ p_amount/);
 assert.doesNotMatch(sql,/\bCOMMIT\b|\bROLLBACK\b/);
 const run={success:true,results:[{id:step.id,tool:step.tool,...output}]};
 const answer=await composeAnswer(task,routed.plan,run);
 assert.equal(answer.success,true);
 assert.equal(answer.source,"tool");
 const validation=await validateResult(task,routed.plan,run,answer);
 assert.equal(validation.valid,true);
});

test("native transfer response is safely Russian even during AI provider 429",async()=>{
 const initial={...safeAccountTransferSqlTool(),mode:"single",source:"tool",tool:"safe_account_transfer_sql"};
 let calls=0;
 const result=await localizeResponse(initial,task,"ru",async()=>{calls++;throw Error("429")});
 const final=enforcePresentationLanguage(result,{task,language:"ru"});
 assert.equal(calls,0);
 assert.equal(final.success,true);
 assert.equal(final.localizationStatus,"already-russian");
 assert.equal(final.text,initial.text);
});
test("unrelated or underspecified transfers are never matched",()=>{
 assert.equal(parseSafeAccountTransferTask("Как перевести деньги?"),null);
 assert.equal(parseSafeAccountTransferTask(task.replace("оба счёта","один счёт")),null);
 assert.equal(parseSafeAccountTransferTask(task.replace("Без COMMIT внутри","Добавь COMMIT")),null);
});
