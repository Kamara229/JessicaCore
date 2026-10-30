/** An explicit backend fix directive without a test must not spend AI quota. */
export function pendingZeroDayExplanationHandoff(message){
  const raw=typeof message==="string"?message:"";
  if(!raw || raw.length>3000)return null;
  const body=raw.split(/\r?\n/u).filter(s=>!/^\s*JESSICA_(?:REPLY|END)_[a-f0-9-]{36}\s*$/iu.test(s)).join(" ").trim();
  if(!/^Результат\s+0\s+верный,\s+но\s+расч[её]т\s+не\s+показан\./iu.test(body) ||
     !/Следующий\s+шаг\s*:\s*исправить\s+объяснение\s+нулевого\s+интервала\s+в\s+dateDiffResultText\.js/iu.test(body) ||
     !/Повторный\s+тест\s+пока\s+не\s+запускай/iu.test(body) ||
     /(?:отправь|выполни|запусти)\s+в\s+обычном\s+чате/iu.test(body))return null;
  return {
    action:"reply",task:"",continueThread:false,
    reply:"Поняла: 0 дней вычислено правильно, но объяснение расчёта не показано. Исправление формата ответа должно быть проверено через CI и Render. Повторный тест пока не запускаю."
  };
}
