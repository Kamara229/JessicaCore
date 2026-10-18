import { sanitizeSourceUrlInput } from "../../tools/sourceUrlInput.js";

/*
 * Bounded native plan for comparing two documents supplied by the user.
 * Resolves the exact supplied URLs without AI; factual comparison and
 * claim validation still require the semantic pipeline.
 */
export function buildNativeSourceComparisonPlan(task) {
    const request=String(task || "").trim();
    if (!/(?:сравн\p{L}*|сопостав\p{L}*|compare|comparison)/iu.test(request)) return null;

    const matches=request.match(/https?:\/\/[^\s<>"')\]]+/gi) || [];
    if (matches.length !== 2) return null;

    const urls=matches.map(sanitizeSourceUrlInput);
    try {
        if (urls.some(item => !["https:","http:"].includes(new URL(item).protocol))) return null;
        if (new URL(urls[0]).href === new URL(urls[1]).href) return null;
    } catch { return null; }

    return {
        intent: "native_two_source_comparison",
        requiresTools: true,
        reasoningSummary: "Получить оба явно заданных источника; затем провести независимое сравнение.",
        evidence: {
            mode: "source_content",
            reason: "Каждый вывод сравнения должен опираться на оба прочитанных документа."
        },
        steps: urls.map((url, index) => ({
            id: "fetch_source_" + (index + 1),
            tool: "web_fetch",
            arguments: {url}
        })),
        metadata: {
            plannerVersion: "native-two-source-v1",
            generatedAt: new Date().toISOString()
        }
    };
}
