/*
 * Translate compact SOURCE 1 / SOURCE 2 references produced by the
 * comparison verifier into the original fetched URLs. The model
 * cannot invent new URLs or evidence: quotations are still checked
 * against the separately fetched original source by the local matcher.
 */
export function resolveClaimSourceAlias(sourceUrl,sourceText) {
    const value=String(sourceUrl || "").trim();
    const label=/^SOURCE\s+(\d+)$/i.exec(value);
    if(!label) return value;
    const entries=[...String(sourceText || "")
        .matchAll(/^SOURCE\s+(\d+)\s*\nURL:\s*(https?:\/\/\S+)/gm)];
    const entry=entries.find(x=>x[1]===label[1]);
    return entry?.[2] || "";
}
