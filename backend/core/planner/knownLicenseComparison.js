import { validatePlan } from "./planValidator.js";

/**
 * Stable reference URLs, not legal facts: evidence must still be
 * fetched and verified on every request.
 */
const LICENSE_REFERENCES = [
    ["Apache-2.0", /\bapache(?:\s+license)?\s*(?:2(?:\.0)?|v2)\b/i, "https://www.apache.org/licenses/LICENSE-2.0.txt"],
    ["MPL-2.0", /(?:mpl|mozilla public license)\s*2(?:\.0)?/i, "https://www.mozilla.org/media/MPL/2.0/index.txt"],
    ["BSD-3-Clause", /\bbsd[\s-]*3[\s-]*clause\b/i, "https://opensource.org/license/bsd-3-clause"],
    ["BSD-2-Clause", /\bbsd[\s-]*2[\s-]*clause\b/i, "https://opensource.org/license/bsd-2-clause"],
    ["MIT", /\bmit(?:\s+license)?\b/i, "https://opensource.org/license/mit"]
];

export function buildKnownLicenseComparisonPlan(task) {
    const text = String(task || "");
    if (!/(?:сравн\p{L}*|сопостав\p{L}*|compare|comparison)/iu.test(text) ||
        /https?:\/\//iu.test(text)) return null;
    const found = LICENSE_REFERENCES.filter(([,pattern]) => pattern.test(text));
    if (found.length !== 2) return null;
    return {
        intent: "native_known_license_comparison",
        requiresTools: true,
        reasoningSummary: "Загрузить два канонических текста; условия определить только из загруженных страниц.",
        evidence: {mode: "source_content", reason: "Нужно проверить оба официальных текста."},
        steps: found.map(([name,,url], index) => ({
            id: "fetch_license_" + (index + 1),
            tool: "web_fetch",
            arguments: {url}
        })),
        metadata: {plannerVersion: "native-license-source-v1", licenseIds: found.map(x=>x[0]),
            generatedAt: new Date().toISOString()}
    };
}


/**
 * The MIT License is a software license, not the university.
 * For a narrow question about missing express patent grants, read the
 * actual MIT license rather than search for a nonexistent "MIT Patent
 * Grant Statement".
 */
export function buildMitPatentQuestionPlan(task) {
    const input=String(task||"");
    if(/https?:\/\//iu.test(input)||
       !/\bmit(?:\s+license)?\b/iu.test(input)||
       !/(?:патент|patent)/iu.test(input)||
       !/(?:отсутств|не\s+(?:доказывает|означает)|absence|without|lack)/iu.test(input)||
       !/(?:грант|licen[sc]e|предоставл|прав|grant|right)/iu.test(input)||
       !/(?:доказыв|означа|обосну|почему|следует|prove|imply|mean)/iu.test(input)||
       /\b(?:apache|bsd|mpl)\b/iu.test(input))return null;
    return {
       intent:"native_mit_patent_question",
       requiresTools:true,
       reasoningSummary:"Прочитать первичный текст MIT License; отсутствие прямой оговорки не устанавливает наличие либо отсутствие иных патентных прав.",
       evidence:{mode:"source_content",reason:"Полный оригинальный текст MIT License необходим для ограниченного вывода."},
       steps:[{id:"fetch_mit_license",tool:"web_fetch",arguments:{url:"https://opensource.org/license/mit"}}],
       metadata:{plannerVersion:"native-mit-primary-v1",generatedAt:new Date().toISOString()}
    };
}

export function tryCreateKnownLicenseComparison({task, planningContext={}}={}) {
    const plan = buildKnownLicenseComparisonPlan(task) || buildMitPatentQuestionPlan(task);
    if (!plan) return {handled:false};
    const result = validatePlan(plan, planningContext, task);
    return result?.success === true
        ? {handled:true,plan,source:"canonical-license-texts"}
        : {handled:false,reason:result?.text || "validation failed"};
}
