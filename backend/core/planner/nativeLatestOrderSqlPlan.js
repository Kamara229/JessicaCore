import "../../tools/latestOrderSqlTool.js";
import {parseLatestOrderSqlTask} from "../native/latestOrderSqlIntent.js";
export function buildNativeLatestOrderSqlPlan(task){
 const p=parseLatestOrderSqlTask(task);
 if(!p)return null;
 return {intent:"native_latest_order_sql",requiresTools:true,
 reasoningSummary:"Детерминированный запрос ROW_NUMBER без AI",
 evidence:{mode:"none",reason:"Известна структура таблицы и правила сортировки."},
 experience:{used:false,source:"native-capability",skills:[]},
 steps:[{id:"native_latest_order_sql",tool:"latest_order_sql",arguments:p,experienceUsed:false}],
 metadata:{plannerVersion:"native-sql-v1",generatedAt:new Date().toISOString()}};
}
