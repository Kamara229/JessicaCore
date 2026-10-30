/** Quota-free acknowledgement of a specific no-test backend repair directive. */
export function pendingCalendarDirectFix(message) {
  const raw=typeof message==="string"?message:"";
  if(!raw || raw.length>3000)return null;
  const body=raw.split(/\r?\n/u)
    .filter(line=>!/^\s*JESSICA_(?:REPLY|END)_[a-f0-9-]{36}\s*$/iu.test(line))
    .join(" ").replace(/\s+/gu," ").trim();
  if(!/^Ответ верный, но Composer всё ещё вызывает AI и получает HTTP 429\./iu.test(body) ||
     !/Следующий шаг: добавить invalid_calendar_date в directAnswer\.js/iu.test(body) ||
     !/Запись в GitHub заблокирована\./iu.test(body) ||
     !/Тест пока не запускать\.?$/iu.test(body) ||
     /(?:в обычном чате Jessica отправь|выполни тест|следующий тест)\s*:/iu.test(body))
     return null;
  return {action:"reply",task:"",continueThread:false,
    reply:"Поняла: проверка даты дала правильный ответ, но AI Composer вызван зря. Нужна публикация прямого ответа инструмента и проверка CI/Render. Тест пока не запускаю."};
}
