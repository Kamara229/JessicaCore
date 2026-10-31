import "../../tools/safeAccountTransferSqlTool.js";
import {parseSafeAccountTransferTask} from "../native/safeAccountTransferIntent.js";
export function buildNativeSafeAccountTransferPlan(task){
 if(!parseSafeAccountTransferTask(task))return null;
 return {intent:"native_safe_account_transfer_sql",requiresTools:true,
 reasoningSummary:"Проверенный PL/pgSQL с порядком блокировок для двух счетов",
 evidence:{mode:"none",reason:"Детерминированная генерация исходного кода без базы данных."},
 experience:{used:false,source:"native-capability",skills:[]},
 steps:[{id:"native_safe_account_transfer",tool:"safe_account_transfer_sql",arguments:{},experienceUsed:false}],
 metadata:{plannerVersion:"native-transfer-v1",generatedAt:new Date().toISOString()}};
}
