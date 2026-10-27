/**
 * Quota-free, non-executing acknowledgement for an already-merged PR report.
 * Treat CI/PR claims as unverified input, never as a completed Jessica test.
 * Android supplies the marker-verified body; other clients may send the
 * complete envelope. This policy grants no tools, execution or test chat.
 */
export function mergedPrProgressHandoff(message) {
  if (typeof message!=="string" || message.length>1100)return null;
  const input=message.trim();
  const start=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})\r?\n/iu.exec(input);
  let raw=input;
  if(start){
    const end="JESSICA_END_"+start[1];
    if(input.split(/\r?\n/u).at(-1)?.trim()!==end)return null;
    raw=input.slice(start[0].length,input.lastIndexOf(end)).trim();
  }else if(/JESSICA_[A-Z]+_/iu.test(input))return null;
  const body=raw.replace(/\s+/gu," ").trim();
  if(!body||body.length>650)return null;
  // Never classify hypothetical, negative, or quoted PR/CI statuses.
  if(!/^PR\s*#\d{1,5}\s+объедин[её]н,\s*CI\s+успешен\./iu.test(body))return null;
  if(/(?:\bCI\s+не\s+успешен|ошибка\s+CI|CI\s+(?:failed|провален)|если\s+PR)/iu.test(body))return null;
  // One technical result and a single, bounded safe backend next step.
  const next=/(?:^|[.!?]\s+)Следующий\s+шаг\s*[—–:-]\s*([^.!?]{8,190})\.?$/iu.exec(body);
  if(!next)return null;
  const step=next[1].trim();
  if(!/^(?:проверить|проверка|добавить|закрепить|исследовать|проанализировать|продолжить|сопоставить|CI[- ]тест)(?:\s|$)/iu.test(step))
    return null;
  if(!/(?:Candidate|Proposal|Skill|PostgreSQL|Supabase|GitHub|Render|backend|\bCI\b|конкурент|транзакц|ревизи|идемпотент|атомарн)/iu.test(step))
    return null;
  const before=body.slice(0,next.index);
  if(!/(?:Skill|Candidate|Proposal|PostgreSQL|Supabase|ACK|worker|backend|\bCI\b)/iu.test(before))
    return null;
  // Explicit Jessica tests, device actions and secrets always need their own
  // authorization path; do not allow this status shortcut to hide them.
  if(/(?:в\s+обычном\s+чате|чат\s+Jessica|запусти|выполни|отправь|создай\s+чат|\bAPK\b|парол|секрет|токен|API[- ]?ключ|настройк|установи|удали|удалить|введи|сервис\s+говорит)/iu.test(body))
    return null;
  return "Поняла: PR и CI отмечены в сообщении, но тест Jessica не выполнялся. "+
    "Проверь статус через GitHub, Render и Supabase и продолжай указанный "+
    "backend-шаг. Тест в обычном чате Jessica пока не запрошен.";
}
