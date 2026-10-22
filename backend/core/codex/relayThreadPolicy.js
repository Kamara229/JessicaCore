/**
 * Test-chat reuse is explicitly opt-in. A new/other chat request overrides
 * even an earlier same-chat phrase. This function never reads private history.
 */
export function explicitSameTestChat(message) {
    const text = String(message || "").replace(/\s+/g," ").trim().toLowerCase();
    if (!text) return false;

    // Explicit isolation/negation beats any conflicting continuation wording.
    const newThread = /(?:нов(?:ый|ом|ую|ого)|отдельн(?:ый|ом|ую)|друг(?:ой|ом|ую))\s+(?:тестов(?:ый|ом|ую)\s+)?чат(?:е|а|у|ом)?|(?:new|separate|different)\s+(?:jessica\s+)?(?:test\s+)?chat/u;
    const denied = /(?:не\s+(?:используй|использовать|продолжай|продолжать|выполняй|выполнять|возвращайся|нужно|надо|следует)[^.!?;]{0,100}чат|не\s+в\s+(?:том|этом|предыдущем)\s+чате|(?:do\s+not|don't|never)\s+(?:use|reuse|continue)[^.!?;]{0,100}chat)/u;
    if (newThread.test(text) || denied.test(text)) return false;

    return /(?:в том же|в этом же|в прежнем|в предыдущем) (?:самом )?(?:тестовом )?чате(?: джессики| jessica)?/.test(text) ||
        /(?:используй|продолжи|продолжай) (?:именно )?(?:тот же|предыдущий) (?:тестовый )?чат(?: джессики| jessica)?/.test(text) ||
        /(?:use|continue in|reuse|in) (?:the )?(?:same|previous) (?:jessica )?(?:test )?chat/.test(text);
}
