/**
 * A ChatGPT conversation-length UI notice is advisory, not an instruction
 * to terminate or open a different thread. Only match a standalone banner,
 * never a user's engineering request which quotes or discusses a banner.
 */
export function isStandaloneLengthNotice(message) {
    if (typeof message !== "string") return false;
    const normalized=message.trim().replace(/\s+/g," ").toLowerCase();
    if (!normalized || normalized.length>500) return false;
    const russian=/^(?:(?:chatgpt|система)\s*[:—-]\s*)?вы достигли максимальн\w* длин\w* (?:этого )?обсуждени\w*(?:[.,:;!? ]|$)/u;
    const english=/^(?:(?:chatgpt|system)\s*[:—-]\s*)?(?:you(?:'ve| have) reached|this conversation has reached) (?:the )?maximum (?:conversation )?length(?:[.,:;!? ]|$)/u;
    return russian.test(normalized) || english.test(normalized);
}

export function lengthNoticeContinuation() {
    return {
        action:"reply",
        reply:"Уведомление ChatGPT о длине обсуждения не завершает Кодекс. Продолжаю работу в выбранном чате без его смены и без повторной отправки.",
        task:"",
        continueThread:false
    };
}
