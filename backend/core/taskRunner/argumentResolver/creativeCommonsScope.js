/*
 * Creative Commons version + variant identity.
 * A comparison can cite only the exact requested licence family/version.
 * No URL is invented: candidates must come from search results.
 */
export function requestedCreativeCommonsVariant(query) {
    const text = String(query || "").trim();
    if (!/(?:creative\s+commons|creativecommons|\bcc\s+by\b)/i.test(text)) return null;
    const match = text.match(/\b(?:cc\s+)?(by(?:[-\s]?sa)?)\s+(\d+(?:\.\d+)?)\b/i);
    if (!match) return null;
    return {
        family: /sa$/i.test(match[1]) ? "by-sa" : "by",
        version: match[2]
    };
}

export function filterCreativeCommonsSources(query, candidates) {
    const variant = requestedCreativeCommonsVariant(query);
    if (!variant) return null;
    const result = Array.isArray(candidates) ? candidates : [];
    const expected = "/licenses/" + variant.family + "/" + variant.version;
    return result.filter(item => {
        try {
            const url = new URL(String(item?.url || ""));
            if (url.protocol !== "https:") return false;
            if (!["creativecommons.org","www.creativecommons.org"].includes(url.hostname.toLowerCase())) return false;
            const path = url.pathname.toLowerCase().replace(/\/+$/, "");
            return path === expected || path.startsWith(expected + "/");
        } catch {
            return false;
        }
    });
}
