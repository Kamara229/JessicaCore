/**
 * Output contract for explicit "ответь только числом" requests.
 * Formatting only: never compute or guess the answer.
 * Unrecognized text remains unchanged so errors stay visible.
 */
const NUM = String.raw`[+\-−]?\d+(?:[.,]\d+)?`;
const PLAIN_NUMBER = new RegExp(`^(${NUM})\\.?$`, "u");
const SIMPLE_EQUATION = new RegExp(
    `^${NUM}\\s*[+×*÷/:−-]\\s*${NUM}\\s*=\\s*(${NUM})\\.?$`,
    "u"
);

function requiresNumericOnly(task) {
    return typeof task === "string" &&
        /(?:^|[.!?]\s+)(?:ответь|напиши|выведи|укажи)\s+(?:(?:строго|ровно)\s+)?только\s+числом\b/iu.test(task);
}

export function applyStrictNumericOnlyFormat(task, response) {
    if (!requiresNumericOnly(task) || response?.success !== true ||
        response?.status !== "COMPLETED" || typeof response?.text !== "string") {
        return response;
    }

    const text = response.text.trim().replace(/[\u00a0\u202f]/gu, " ");
    const match = PLAIN_NUMBER.exec(text) || SIMPLE_EQUATION.exec(text);
    if (!match) return response;

    return { ...response, text: match[1] };
}
