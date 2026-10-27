/**
 * Interpret a bounded, no-test backend incident update without AI tokens.
 * A status claim is not proof of CI, deployment, or an executed Jessica test.
 */
export function incidentCiStatusHandoff(message) {
  if(typeof message!=="string"||message.length>1200)return null;
  const input=message.trim();
  const start=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\r?\n/iu.exec(input);
  let source=input;
  if(start){
    const end="JESSICA_END_"+start[1];
    if(input.split(/\r?\n/u).at(-1)?.trim()!==end)return null;
    source=input.slice(start[0].length,input.lastIndexOf(end));
  }else if(/JESSICA_[A-Z]+_/iu.test(input))return null;
  const body=source.replace(/\s+/gu," ").trim();
  if(!body||body.length>750)return null;
  // Require actual incident, linked merged-code fix, CI and explicit no-test.
  if(!/^Причина\s+сбоя\s*:/iu.test(body))return null;
  if(!/\bPR\s*#\d{1,5}\s+не\s+распознавал(?:ся|ась|ось|ись)\s+без\s+ИИ[.!?]/iu.test(body))return null;
  if(!/Исправлен[оа]?\s+в\s+PR\s*#\d{1,5}\s*,\s*CI\s+прош[её]л\./iu.test(body))return null;
  if(!/(?:^|[.!?]\s+)Тест\s+Jessica\s+не\s+запускался\.?$/iu.test(body))return null;
  const next=/(?:^|[.!?]\s+)Далее\s*[—–:-]\s*([^.!?]{8,180})\.[ ]+Тест\s+Jessica\s+не\s+запускался\.?$/iu.exec(body);
  if(!next)return null;
  const step=next[1].trim();
  if(!/^(?:проверить|проверка|продолжить|исследовать|проанализировать|сопоставить|закрепить|добавить)(?:\s|$)/iu.test(step))return null;
  if(!/(?:\bCI\b|\bPR\b|backend|GitHub|Render|Supabase|Candidate|Proposal|Skill|PostgreSQL|конкурент|гонк|транзакц)/iu.test(step))return null;
  if(/(?:если|возможно|предположительно|планируется|не\s+исправлен|CI\s+не\s+прош[её]л|ошибка\s+CI|CI\s+failed)/iu.test(body))return null;
  if(/(?:в\s+обычном\s+чате|создай\s+чат|отправь|запусти|выполни|повтори|\bAPK\b|парол|секрет|токен|API[- ]?ключ|настройк|установи|удали|удалить|введи)/iu.test(body))return null;
  return "Поняла сообщение об исправлении сбоя интерпретации. "+
    "Это не выполненный тест Jessica и не доказательство развёртывания. "+
    "Проверь GitHub CI, Render и Supabase, затем продолжай текущий backend-шаг. "+
    "Тест в обычном чате Jessica пока не запрошен.";
}
