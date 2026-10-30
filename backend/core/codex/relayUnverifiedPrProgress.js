/** Provider-free handoff for a failed CI/PR with an explicit do-not-test instruction. */
export function unverifiedPrProgress(message){
 if(typeof message!=="string"||message.length>10000)return null;
 const body=message.split(/\r?\n/u).filter(x=>
  !/^\s*JESSICA_(?:REPLY|END)_[\da-f-]{36}\s*$/iu.test(x)
 ).join(" ").replace(/\s+/gu," ").trim();
 if(!/\bPR\s*#?\d{1,6}\b/iu.test(body) ||
   !/\bCI\s+(?:не\s+пройд[её]н|провал[её]н|failed)/iu.test(body) ||
   !/(?:повторн(?:ый|ого)\s+тест|тест)\s+пока\s+не\s+(?:запускай|повторяй)/iu.test(body))return null;
 if(/(?:отправь|запусти|выполни)\s+в\s+обычн/iu.test(body)||
   /[«»]/u.test(body))return null;
 return "Поняла: исправление ещё не прошло CI, а повторный тест запрещён. Проверь упавшие проверки GitHub, исправь backend и подтверди развёртывание в Render. Supabase не изменяю; успешного результата теста нет.";
}
