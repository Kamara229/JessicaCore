/**
 * CALIBRI S28+ progress (including S30 FP32 TRAIN skill regression) is a
 * research instruction for main ChatGPT/GitHub, not a Jessica ordinary-chat
 * test. This is the fail-closed provider-free path for the 2026-11-08 outage.
 *
 * No external tools, GitHub calls, training or QA are performed here.
 * Never trust unmatched markers, ambiguous follow-up, secrets or test requests.
 */
export function calibriLaterResearchNoJessicaTest(message) {
    if (typeof message!=="string" || !message.trim() || message.length>1500) return null;
    const lines=message.trim().split(/\r?\n/u).map(x=>x.trim()).filter(Boolean);
    const match=/^JESSICA_REPLY_([\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12})$/iu.exec(lines[0]||"");
    let body;
    if(match){
        if(lines.length<3 || lines.at(-1)!=="JESSICA_END_"+match[1])return null;
        body=lines.slice(1,-1).join(" ");
    }else{
        if(/JESSICA_(?:REPLY|END)_/iu.test(message))return null;
        body=lines.join(" ");
    }
    body=body.replace(/\s+/gu," ").trim();
    if(!body || body.length>800)return null;
    // S41 research-only backend incident report is not a command to
    // Jessica; do not send it to a quota-limited interpretation provider.
    // Exact fully bounded text prevents swallowing a real test or action.
    const statusS41="Render: сбой интерпретации; Supabase исправна. "+
      "Обработчик не узнавал «Шаг S41». Исправлено в draft PR #418, "+
      "CI пройден; в LIVE пока не внедрено. Исследование S41 — draft "+
      "PR #425, CI #42498851459. Jessica не тестировать.";
    if(body===statusS41) return {
       action:"reply",task:"",continueThread:false,
       reply:"Поняла: S41 — отдельный GitHub CI. Исправление интерпретатора находится "+
             "только в исследовательском PR, в LIVE не внедрено. "+
             "Jessica не выполняет тесты. Продолжаем план через GitHub."
    };
    // S42 regression: an artifact card about 64 S41 TRAIN items is a
    // passive GitHub-research handoff, never an ordinary Jessica test.
    // Match only the exact known benign data citation; no broad URL allowance.
    const s41AnswerArtifact =
      "S41: [64 TRAIN до/после](sandbox:/mnt/data/CALIBRI_S41_64_TRAIN_answers.json). "+
      "Шаг S42 — сравнить predicted_winner_token, noun_gold_rank и generated_text: "+
      "после S41 лидируют «ж», «dor», «red». Jessica не тестировать, LIVE не менять.";
    if(body===s41AnswerArtifact) return {
      action:"reply",task:"",continueThread:false,
      reply:"Получен исследовательский отчёт S41 о 64 TRAIN-ответах. "+
            "Анализ S42 продолжает ChatGPT через GitHub и не через Jessica. "+
            "Тест в обычном чате не запрашивался и не запускался. LIVE не меняю."
    };
    // Stable category route, NOT one new hardcoded sentence per outage:
    // GitHub-only CALIBRI research progress may start with Render or CI
    // and still explicitly forbid Jessica ordinary-chat tests.
    // Fail closed on anything resembling an action, deployment or secrets.
    const protectedCalibrIResearch =
      /(?:^|[^A-Za-z\d])S(?:2[89]|[3-9]\d|1\d\d)(?=[^A-Za-z\d]|$)/iu.test(body) &&
      /(?:GitHub|CI\s*#\d{6,}|draft\s+PR\s*#\d+|TRAIN)/iu.test(body) &&
      /Следующ(?:ий|ая)\s+шаг(?:\s+ChatGPT)?\s*[—–:-]\s*(?:проверить|изучить|сравнить|проанализировать|исследовать|обучать|тренировать|устранить|исправить)(?=\s|[.!?])/iu.test(body) &&
      /(?:Jessica|Джессика)\s+не\s+тестировать(?:\s*,\s*LIVE\s+не\s+менять)?\s*[.!]?\s*$/iu.test(body) &&
      /(?:LIVE\s+не\s+(?:измен[её]н|изменял[аи]?|менять|менял[аи]?|меняли|обновл[её]н|трогать)|в\s+LIVE\s+(?:не\s+внедрено|пока\s+не\s+внедрено))/iu.test(body);
    // Any competing instruction always defeats provider-free classification.
    const unsafeOrAmbiguousResearch =
      /(?:создай|выполни|запусти|проведи|повтори|отправь|удали|измени|обнови|разверни|выпусти|установи|протестируй)\s+(?:(?:новый|один|ещё|этот|следующий)\s+)?(?:тест|тестовый|чат|задач|деплой|deploy|сервер|базу|модель|кодекс)/iu.test(body) ||
      /(?:Jessica|Джессика)[,:]?\s+(?:проведи|запусти|протестируй|выполни|открой|создай)/iu.test(body) ||
      /(?:тест|проверка)\s+(?:в\s+)?(?:Jessica|Джессика)\s+(?:успешен|пройден|выполнен|заверш[её]н)/iu.test(body) ||
      /(?:в\s+обычном\s+чате|тест\s+в\s+Jessica|через\s+Jessica|точная\s+(?:формулировка|задача)|API[- ]?ключ|парол|секрет|токен\s+доступа|deploy|merge|https?:\/\/|sandbox:|[<>{}`])/iu.test(body) ||
      /(?:LIVE|Render|Supabase|APK)\s*(?:изменить|обновить|удалить|перезапустить|деплоить|задеплоить)/iu.test(body) ||
      /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body);
    if(protectedCalibrIResearch && !unsafeOrAmbiguousResearch) {
      const candidates = [...body.matchAll(/(?:^|[^A-Za-z\d])S(2[89]|[3-9]\d|1\d\d)(?=[^A-Za-z\d]|$)/giu)];
      const number = candidates.length ? candidates.at(-1)[1] : "?";
      return {
        action:"reply",task:"",continueThread:false,
        reply:"Поняла: CALIBRI S"+number+". Проверки и исследование ведёт ChatGPT через GitHub. "+
              "Jessica тестов не выполняет. Сведения о CI не означают успешного "+
              "обычного теста Jessica. LIVE не меняю."
      };
    }
    const version=/^(?:CALIBRI\s+)?S(2[89]|[3-9]\d|1\d\d)(?=\s|[:.,;—–-])/iu.exec(body);
    if(!version)return null;
    // S38 incident: owner gave "Следующий шаг ChatGPT — S39" and
    // "Jessica не тестировать, LIVE не менять". Older parser required the
    // phrase "Jessica тесты не запускает" at the very end; an explicit
    // protected GitHub-only research step fell into quota-limited Groq.
    const legacyNoTest=/(?:Jessica|Джессика)\s+тесты?\s+не\s+запуска(?:ет|ю|й|ть)\s*[.!]?$/iu.test(body);
    const researchOnlyNoTest=/(?:Jessica|Джессика)\s+не\s+тестировать\s*,\s*LIVE\s+не\s+меня(?:ть|й)(?=\s|[.!?]|$)/iu.test(body);
    // S40 incident: a short research-only instruction starts "Шаг S41:"
    // instead of "Следующий шаг ChatGPT —". The old strict phrase guard
    // missed it and needlessly called the rate-limited interpreter.
    const hasResearchStep =
        /Следующ(?:ий|ая)\s+шаг(?:\s+ChatGPT)?\s*[—–:-]\s*\S+/iu.test(body) ||
        /(?:^|[.!?]\s+)Шаг\s+S(?:2[89]|[3-9]\d|1\d\d)\s+для\s+ChatGPT\s*:\s*(?:тренировать|обучать|устранить|проверить|сравнить|исследовать|проанализировать|изучить|улучшить)(?=\s|[.!?])/iu.test(body) ||
        /(?:^|[.!?]\s+)Шаг\s+S(?:2[89]|[3-9]\d|1\d\d)\s*:\s*(?:тренировать|обучать|диагностир\w*|сравни\w*|исследов\w*|проверить|проанализиров\w*|изучить|улучш\w*)/iu.test(body);
    if(!hasResearchStep ||
       !/(?:FP32|TRAIN|Green Tunnel|bpp|QA|навык|обучени|обучать|исследован|сжатие|кодовые\s+книги)/iu.test(body) ||
       !(legacyNoTest || researchOnlyNoTest))
       return null;
    // A status note must not accidentally suppress an actual Jessica chat test,
    // credential request, external action or an explicitly finished plan.
    if(/(?:создай|открой|выполни|запусти|проведи|повтори|отправь|протестируй)\s+(?:(?:один|этот|новый|следующий)\s+)?(?:тест(?:ы)?|тестовый\s+чат|задач[уа])/iu.test(body) ||
       /(?:в\s+обычном\s+чате|через\s+обычный\s+чат|Точная\s+(?:формулировка|задача|инструкция)|парол|секрет|API[- ]?ключ|токен\s+(?:доступа|авторизац\w*|GitHub|API)|удали|установи|deploy|merge|https?:\/\/|[\x60«»])/iu.test(body) ||
       /(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н/iu.test(body))
       return null;
    return {
      action:"reply",task:"",continueThread:false,
      reply:"Поняла: CALIBRI S"+version[1]+". Следующий исследовательский шаг "+
            "выполняет ChatGPT через GitHub, не Jessica. "+
            "Тест в обычном чате не запрашивался и не запускался. "+
            "Продолжайте работу с фактическими результатами, без домыслов."
    };
}
