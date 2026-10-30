export function workedAnswerNoTest(message) {
  if (typeof message !== "string" || message.length > 3000) return null;
  const body=message.trim().split("\n");
  if (body.length<4 || !/^JESSICA_REPLY_[a-f0-9-]{36}$/iu.test(body[0].trim()) ||
      !/^JESSICA_END_[a-f0-9-]{36}$/iu.test(body.at(-1).trim())) return null;
  if (body[0].trim().slice(14)!==body.at(-1).trim().slice(12)) return null;
  const lines=body.slice(1,-1).map(x=>x.trim()).filter(Boolean);
  if (lines.length<2 || lines.length>12) return null;
  if (!lines.every(x=>/^[\d\s()+\-×*=]+$/u.test(x))) return null;
  if (lines.filter(x=>x.includes("=")).length<2) return null;
  return {action:"reply",task:"",continueThread:false,
    reply:"Ответ ChatGPT содержит готовый расчёт, но не команду выполнить тест в обычном чате Jessica. Тест не запускался, результата Jessica нет."};
}