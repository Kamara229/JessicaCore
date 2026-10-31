import test from "node:test";
import assert from "node:assert/strict";
import {parseLatestOrderSqlTask} from "../core/native/latestOrderSqlIntent.js";
import {decomposeTask} from "../core/taskDecomposer.js";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";
import {latestOrderSqlTool} from "../tools/latestOrderSqlTool.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
const task="Есть orders(id,user_id,created_at,amount). Напиши SQL для получения последнего заказа каждого клиента. При одинаковой дате выбирай больший id. Объясни решение.";

test("latest-order SQL is routed without AI, including real decomposition",async()=>{
 assert.deepEqual(parseLatestOrderSqlTask(task),{explain:true});
 const d=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(d.native,true);
 assert.equal(d.decomposition.subtasks[0].text,task);
 const routed=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(routed.handled,true);
 assert.equal(routed.plan.intent,"native_latest_order_sql");
 const step=routed.plan.steps[0];
 const tool=latestOrderSqlTool(step.arguments);
 assert.equal(tool.success,true);
 assert.match(tool.text,/PARTITION BY user_id ORDER BY created_at DESC, id DESC/u);
 assert.match(tool.text,/WHERE rn = 1/u);
 const run={success:true,results:[{id:step.id,tool:step.tool,...tool}]};
 const answer=await composeAnswer(task,routed.plan,run);
 assert.equal(answer.success,true);
 assert.equal(answer.source,"tool");
 assert.equal(answer.text,tool.text);
 const validation=await validateResult(task,routed.plan,run,answer);
 assert.equal(validation.valid,true);
});
test("incomplete latest order requests are not assumed",()=>{
 assert.equal(parseLatestOrderSqlTask(task.replace("больший id","случайный id")),null);
 assert.equal(parseLatestOrderSqlTask(task.replace("created_at,","")),null);
});
