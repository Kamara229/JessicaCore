import { matchesAttributedEvidence } from "../validator/claim/claimEvidenceMatcher.js";
import { createComparisonEvidenceLedger, renderComparisonEvidenceLedger } from "./comparisonEvidenceLedger.js";

/*
 * A bounded user-facing fallback after comparison presentation failure.
 * Claims marked "supported" have source-attributed verbatim quotes, but
 * their semantic interpretation can still be wrong. Show quotations only.
 * Never turn this into a fully verified answer or positive Experience.
 */
function sourceKey(value) {
    try {
        const url = new URL(String(value || ""));
        return url.hostname.toLowerCase() + url.pathname.replace(/\/+$/, "").toLowerCase();
    } catch {
        return "";
    }
}
function safeExcerpt(value) {
    return String(value || "").replace(/\s+/g, " ").trim().slice(0, 190);
}
export function buildPartialComparisonEvidence(task, plan, runResult, claims) {
    if (!/(?:сравн|сопостав|compar)/iu.test(String(task || ""))
        || plan?.evidence?.mode !== "source_content") return null;

    const fetched=(runResult?.results || []).filter(item =>
        item?.tool === "web_fetch" && item.success === true
        && item.data?.url && typeof item.data?.content === "string"
        && item.data.content.trim()
    );
    if (fetched.length !== 2) return null;
    const urls=fetched.map(item=>sourceKey(item.data.url));
    if (!urls[0] || !urls[1] || urls[0]===urls[1]) return null;

    const excerpts=[[], []];
    for (const claim of Array.isArray(claims) ? claims : []) {
        if (claim?.supported !== true) continue;
        const index=urls.indexOf(sourceKey(claim.sourceUrl));
        if (index<0) continue;
        const quote=safeExcerpt(claim.evidence);
        if (quote.length<20 || quote.split(" ").length<4) continue;
        const source=fetched[index];
        const sourceText="SOURCE 1\nURL: "+source.data.url+"\nSNIPPET: "+source.data.content;
        if (!matchesAttributedEvidence({sourceUrl:source.data.url,evidence:quote},sourceText)) continue;
        if (!excerpts[index].includes(quote)) excerpts[index].push(quote);
    }
    if (excerpts.some(items=>items.length<2)) return null;

    // When the originals contain enough topical evidence, show a compact
    // criterion-by-criterion table, rather than an unsorted list of quotes.
    // This remains degraded: extracted quotations do NOT prove semantics.
    const sourceLedger=renderComparisonEvidenceLedger(
        createComparisonEvidenceLedger(task,runResult)
    );
    if (sourceLedger) {
        return {text:sourceLedger.text, quoteCounts:excerpts.map(items=>items.length),
            rows:sourceLedger.rows, degraded:true};
    }

    const result=[
        "Частичный результат анализа документов",
        "",
        "Полный сравнительный вывод не прошёл проверку. Ниже — дословные выдержки, найденные в соответствующих первоисточниках. Их содержание требует самостоятельной интерпретации.",
        "",
        "Источник 1: "+fetched[0].data.url,
        ...excerpts[0].slice(0,5).map(quote=>"• «"+quote+"»"),
        "",
        "Источник 2: "+fetched[1].data.url,
        ...excerpts[1].slice(0,5).map(quote=>"• «"+quote+"»"),
        "",
        "Ограничение: это не полное сравнение лицензий. Неподтверждённые выводы исключены."
    ].join("\n");
    return {text:result, quoteCounts:excerpts.map(items=>items.length)};
}
