export function parseLatestOrderSqlTask(task) {
 const t=String(task||"");
 if(t.length>900||!t.trim())return null;
 if(!/orders\s*\(\s*id\s*,\s*user_id\s*,\s*created_at\s*,\s*amount\s*\)/iu.test(t))return null;
 if(!/SQL/iu.test(t)||!/(последн|latest)/iu.test(t)||!/(кажд|every)/iu.test(t)||!/(клиент|client|user)/iu.test(t))return null;
 if(!/(одинаков|same)/iu.test(t)||!/(дат|date)/iu.test(t)||!/(больш|higher)/iu.test(t)||!/(\bid\b)/iu.test(t))return null;
 return {explain:/(объясн|explain)/iu.test(t)};
}
