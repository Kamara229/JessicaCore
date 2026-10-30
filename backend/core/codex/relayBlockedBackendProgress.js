/** Explicit blocked backend handoff: never infer that a test ran or succeeded. */
export function blockedBackendProgress(message) {
 const raw=typeof message==="string"?message:"";
 if(!raw || raw.length>10000)return null;
 const body=raw.split(/\r?\n/u).filter(line=>
   !/^\s*JESSICA_(?:REPLY|END)_[0-9a-f-]{36}\s*$/iu.test(line)
 ).join(" ").replace(/\s+/gu," ").trim();
 if(!/^Тест\s+не\s+пройден\s*:/iu.test(body) ||
   !/GitHub\s+заблокирован[ао]|запись\s+исправления\s+в\s+GitHub\s+заблокирована/iu.test(body) ||
   !/повторн(?:ый|ого)\s+тест\s+пока\s+не\s+(?:запускай|повторяй)/iu.test(body))return null;
 if(/[«»]/u.test(body) ||
   /(?:отправь|запусти|выполни)\s+в\s+обычн/iu.test(body))return null;
 return "Поняла: тест не пройден, запись исправления в GitHub заблокирована. Продолжи диагностику и внесение исправления через подключения; после CI проверь Render. Повторный тест пока не запускаю, успешного результата нет.";
}
