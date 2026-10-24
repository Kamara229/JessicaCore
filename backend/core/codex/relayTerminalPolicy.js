/**
 * A language-model suggestion is not authority to end an owner's active
 * Codex session. Only an explicit finished-plan announcement may end it;
 * the phone's user Stop action remains independent of this policy.
 *
 * Never print or persist ChatGPT message content here.
 */
export function confirmedPlanCompletion(message) {
    if (typeof message !== "string") return false;
    const lines=message.trim().split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
    const first=lines.find(line=>!/^JESSICA_(?:REPLY|END)_/u.test(line)) || "";
    if (/^(?:если|когда|после того как|проверь|исправь|продолжи|следующ)/iu.test(first)) return false;
    return /^(?:(?:итог|результат)[:.!]?\s*)?(?:план (?:на\s+сегодня\s+|работ\s+|улучшений\s+|разработки\s+)?(?:полностью )?(?:заверш[ёе]н|выполнен)|все (?:запланированные )?(?:задачи|проверки) (?:выполнены|завершены))(?:[.!,\s]|$)/iu.test(first);
}

export function guardRelayTerminalDecision(value, message) {
    if (!value || (value.action !== "pause" && value.action !== "finish")) return value;
    if (value.action === "finish" && confirmedPlanCompletion(message)) return value;
    // A model may mistakenly choose pause after a safe text-only test or
    // misread a system length notice as a shutdown request. Neither can
    // turn the already-running phone session off.
    return {
        ...value,
        action:"reply",
        reply:"Работающий Кодекс остаётся в выбранном чате. Продолжай текущий план и предложи следующий безопасный текстовый шаг или один тест."
    };
}
