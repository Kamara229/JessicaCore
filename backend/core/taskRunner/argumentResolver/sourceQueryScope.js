/*
 * Query-scoped source selection for multi-document research.
 * Pure functions: no provider calls or persistence.
 * A fetch for a named license must not silently open the other license.
 */
const normalize = value => String(value || "").trim();

export function resolveReferencedSearchResult(reference, results) {
    if (!Array.isArray(results)) return null;
    const from = normalize(reference?.$from);
    if (from) {
        const match = results.find(x => x?.id === from);
        return match?.tool === "web_search" && match?.success === true
            ? match : null;
    }
    return [...results].reverse().find(x =>
        x?.tool === "web_search" && x?.success === true
    ) || null;
}

function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function scopeCandidatesToSearchQuery(query, candidates) {
    const results = Array.isArray(candidates) ? candidates : [];
    const text = normalize(query);

    // Narrow only when the search explicitly names one license.
    // Other research searches retain the normal Source Selector behavior.
    const match = text.match(/\b([a-z][a-z0-9+._-]{1,40})\s+(?:license|licence)\b(?:\s+(\d+(?:[.-]\d+)?))?/i);
    if (!match) return results;

    const name = match[1];
    if (/^(?:open|source|software|free|public|copyright|creative)$/i.test(name)) {
        return results;
    }

    const token = new RegExp("(^|[^a-z0-9])" + escapeRegex(name) + "($|[^a-z0-9])", "i");
    const version = match[2] || "";
    const versionPattern = version
        ? new RegExp("(^|[^0-9])" + escapeRegex(version).replace(/\\[.-]/g, "[.-]") + "($|[^0-9])", "i")
        : null;

    return results.filter(item => {
        const title = normalize(item?.title);
        let path = "";
        try { path = new URL(normalize(item?.url)).pathname; } catch {}
        const label = title + " " + path;
        return token.test(label) && (!versionPattern || versionPattern.test(label));
    });
}

export function selectionContextForSearch(context, searchResult) {
    const query = normalize(searchResult?.arguments?.query);
    const base = context && typeof context === "object" ? context : {task:normalize(context)};
    return {...base, task: query || normalize(base.task)};
}
