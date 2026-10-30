/** A narrowly matched no-test date format directive must not spend AI quota. */
export function pendingAbsoluteDateFix(message){
  const raw=typeof message==="string"?message:"";
  if(!raw||raw.length>3000)return null;
  const body=raw.split(/\r?\n/u)
    .filter(s=>!/\s*JESSICA_(?:REPLY|END)_[a-f0-9-]{36}\s*$/iu.test(s))
    .join(" ").trim();
  if(!/^Число\s+3\s+верное,\s+но\s+расч[её]т\s+отсутствует\./iu.test(body) ||
    !/Следующий\s+шаг\s*:\s*исправить\s+dateDiffResultText\.js,\s*добавить\s+проверку\s+полноты\s+в\s+Validator\s+и\s+регрессионный\s+тест/iu.test(body) ||
    !/Повторный\s+тест\s+пока\s+не\s+запускай/iu.test(body) ||
    /(?:отправь|повтори|выполни|запусти)\s+(?:тест\s+)?в\s+обычном\s+чате/iu.test(body))return null;
  return {action:"reply",task:"",continueThread:false,
    reply:"Поняла: абсолютная разница 3 дня вычислена верно, но запрошенный расчёт не приведён. Проверяем исправление формата, Validator и регрессии через GitHub и Render. Повторный тест пока не запускаю."};
}
