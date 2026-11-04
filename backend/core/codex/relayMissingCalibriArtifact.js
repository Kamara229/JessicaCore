/**
 * Read-only, no-test handoff for the specific CALIBRI v108 private-artifact
 * audit request when the user explicitly instructs us NOT to invent results
 * if raw model outputs are unavailable to Jessica.
 *
 * The Codex relay interpreter has no GitHub/Render/Supabase tools and cannot
 * inspect original research artifacts. Public web_search is NOT a substitute
 * for private run artifacts; repeated source-selection/replanning here has
 * previously exhausted the 90-second ordinary-chat deadline.
 *
 * Never claim this branch performs the requested fact check. It only
 * communicates the missing evidence to the selected ChatGPT chat.
 */
export function missingCalibriArtifactHandoff(message) {
    if (typeof message !== "string" || !message.trim() || message.length > 1500)
        return null;
    const lines=message.trim().split(/\r?\n/u).map(s=>s.trim()).filter(Boolean);
    let body;
    const open=/^JESSICA_REPLY_([a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12})$/iu.exec(lines[0]||"");
    if (open) {
        if (lines.at(-1)!=="JESSICA_END_"+open[1] || lines.length!==3)
            return null;
        body=lines[1];
    } else {
        if (/JESSICA_(?:REPLY|END)_/iu.test(message) || lines.length!==1)
            return null;
        body=lines[0];
    }
    // Not a generic "tests are optional" rule. Only this bounded
    // inaccessible ORIGINAL v108 research-evidence request is eligible.
    if (!/^Проверь\s+27\s+ответов\s+v108\s+вручную\s*:/iu.test(body) ||
        !/\bFP32\s*[—–-]\s*4\s*\/\s*6\s+верных/iu.test(body) ||
        !/\bCB59\s+и\s+v102\s*[—–-]\s*0\s*\/\s*6/iu.test(body) ||
        !/Если\s+исходные\s+ответы\s+недоступны,\s*сообщи\s+об\s+этом/iu.test(body) ||
        !/Не\s+подтверждай\s+результаты\s+без\s+фактической\s+проверки/iu.test(body))
        return null;
    if (/(?:```|actual_generated_text|generated_token_ids|https?:\/\/|(?:выполни|запусти|повтори)\s+(?:новый\s+)?тест)/iu.test(body))
        return null;
    return {
        action:"reply",
        reply:"Поняла: исходные 27 генераций v108 и детектор мне недоступны. " +
            "Поэтому я не подтверждаю оценки FP32 4/6 и CB59/v102 0/6 без исходных ответов. " +
            "Тест в обычном чате Jessica не запускала. Проверь GitHub Actions " +
            "и продолжи анализ в выбранном чате ChatGPT.",
        task:"",
        continueThread:false
    };
}
