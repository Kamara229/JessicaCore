/** Research-only S88→S89 no-test handoff; no model calls or LIVE actions.
 *
 * The exact completed S88 report (456 unchanged TRAIN, speed, bpp) is an
 * instruction for ChatGPT to continue GitHub research, not an invitation to
 * launch a Jessica chat test. The relay cannot independently verify the CI.
 * Fail closed for malformed envelopes, appended instructions and requests
 * to use secrets, deploy or create tests.
 */
export function calibriS88ToS89NoTest(message) {
  if (typeof message !== "string" || !message.trim() || message.length > 1400)
    return null;
  const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
  const open=/^JESSICA_REPLY_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/iu.exec(lines[0]||"");
  let body;
  if (open) {
    if (lines.length<3 || lines.at(-1)!=="JESSICA_END_"+open[1]) return null;
    body=lines.slice(1,-1).join(" ");
  } else {
    if (/JESSICA_(?:REPLY|END)_/iu.test(message)) return null;
    body=lines.join(" ");
  }
  body=body.replace(/\s+/gu," ").trim();
  if (body.length>600) return null;
  const completedS88 = /^CALIBRI\s+S88\s*:\s*CI\s+успешен\s*[!.]/iu.test(body) &&
      /\b456\s+ответов\s+без\s+изменений(?=\s|[.,!?:;]|$)/iu.test(body) &&
      /\bCPU\s+1[,.]070\s*[×xх]/iu.test(body) &&
      /\bbpp\s+0[,.]6793\b/iu.test(body) &&
      /\bPR\s*#476\s+обновл[её]н(?=\s|[.,!?:;]|$)/iu.test(body) &&
      /Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*S89\s*:\s*проверить\s+точный\s+паритет\s+CPU\s*\/\s*MPS/iu.test(body) &&
      /(?:Jessica|Джессика)\s+не\s+тестировать\s*,\s*LIVE\s+не\s+менять\s*[.!]?$/iu.test(body);
  // After a source-only fix PR is reported, the next turn remains a passive
  // handoff: "CI успешен" is NOT proof that the backend deployed to Render.
  const backendFixPR = /^В\s+PR\s*#477\s+исправлена\s+обработка\s+S88\s*[→➝-]\s*S89\s+без\s+AI[- ]провайдера\s*;/iu.test(body) &&
      /\bCI\s+успешен(?=[.!?\s]|$)/iu.test(body) &&
      /Supabase\s+работает/iu.test(body) &&
      /Render\s+требует\s+подтверждения\s+workspace/iu.test(body) &&
      /Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*S89\s*:\s*проверить\s+паритет\s+CPU\s*\/\s*MPS\s+через\s+GitHub/iu.test(body) &&
      /LIVE\s+не\s+менять\s*,\s*(?:Jessica|Джессика)\s+не\s+тестировать\s*[.!]?$/iu.test(body);
  if (!completedS88 && !backendFixPR) return null;
  if (/(?:запусти|проведи|выполни|повтори|создай|открой|отправь)\s+(?:(?:новый|один|следующий)\s+)?(?:тест(?:ы)?|тестовый\s+чат|задач[уа])/iu.test(body) ||
      /(?:парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац)|deploy|merge|разверни|установи|удали|https?:\/\/|[`«»])/iu.test(body))
    return null;
  return {
    action:"reply",task:"",continueThread:false,
    reply: backendFixPR
      ? "Поняла: PR #477 исправляет исходный код обработки сообщения S88→S89, но это ещё не подтверждённый LIVE-деплой. Исследование CPU/MPS продолжает ChatGPT через GitHub. Тест Jessica не запускался."
      : "Поняла: ChatGPT сообщил об успешном CI S88 и следующем исследовательском шаге S89. Проверку CPU/MPS выполняет ChatGPT через GitHub; самостоятельно CI не подтверждаю. Тест Jessica не запрошен и не запускался, LIVE не меняю."
  };
}
