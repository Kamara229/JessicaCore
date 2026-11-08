/**
 * Narrow provider-free CALIBRI quota incident handoff.
 *
 * This is a relay status, not an instruction to run a Jessica chat test.
 * It matches an intact JESSICA envelope, a specific quota diagnostic,
 * and the existing "LIVE not changed / native C in GitHub" boundary.
 * Any extra action, secret, chat test, URL or malformed marker falls
 * through to the existing interpreter (no invented successful result).
 */
export function codexQuotaIncidentNoTestStatus(message) {
  if(typeof message!=="string" || message.length>1050)return null;
  const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
  if(lines.length!==3)return null;
  const marker=/^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/iu.exec(lines[0]);
  if(!marker || lines[2]!=="JESSICA_END_"+marker[1])return null;
  const body=lines[1];
  if(body.length>380 || !/^Render:\s*Groq\s+429\s*[—–-]/iu.test(body))return null;
  if(!/суточный\s+лимит\s+200\s*тыс\.\s*токенов/iu.test(body))return null;
  if(!/Supabase\s+исправен/iu.test(body))return null;
  if(!/research\s+PR\s+#\d+/iu.test(body))return null;
  if(!/обход\s+AI\s+для\s+S62\s+без\s+теста\s+Jessica/iu.test(body))return null;
  if(!/LIVE\s+не\s+менял/iu.test(body))return null;
  if(!/Далее\s*[—–-]\s*native\s+C\s+в\s+GitHub\.?$/iu.test(body))return null;
  if(/(?:запусти|повтори|выполни|проведи|отправь|создай|установи|измени|deploy|merge|удали|секрет|парол|ключ|https?:|[«»`])/iu.test(body))return null;
  return {action:"reply",task:"",continueThread:false,
    reply:"Сбой Groq 429 подтверждён; тест Jessica не запускался. "+
      "Исследовательскую проверку S62 native C ведёт ChatGPT в GitHub. "+
      "LIVE Jessica, Render и Supabase не изменяю. Суточную квоту Groq кодом не увеличиваю."};
}
