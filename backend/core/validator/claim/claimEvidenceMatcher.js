/*
 * Ground quoted evidence in the attributed source, rather than in
 * any of the other documents included in a combined comparison prompt.
 *
 * Fetch returns Markdown. Formatting-only differences (links, emphasis,
 * punctuation, whitespace) are ignored, but word order and vocabulary
 * must remain a contiguous match to the original source.
 */
function normalizeWords(value) {
    return String(value || "")
        .normalize("NFKC")
        .replace(/\[([^\]]+)\]\(\s*https?:\/\/[^)]+\)/g, "$1")
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;|&#160;/gi, " ")
        .replace(/&amp;/gi, "&")
        .replace(/[\p{P}\p{S}]+/gu, " ")
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim();
}

function canonical(urlValue) {
    try {
        const url=new URL(String(urlValue || ""));
        if(!["https:","http:"].includes(url.protocol)) return "";
        return url.hostname.toLowerCase()+url.pathname.replace(/\/+$/,"").toLowerCase();
    } catch {
        return "";
    }
}

function sourceSections(sourceText) {
    const text=String(sourceText || "");
    const starts=[...text.matchAll(/^SOURCE\s+\d+\s*$/gm)];
    if (!starts.length) return null;
    return starts.map((item,index)=>{
        const start=item.index;
        const end=index+1<starts.length?starts[index+1].index:text.length;
        const segment=text.slice(start,end);
        const url=segment.match(/^URL:\s*(\S+)/m);
        return {url:canonical(url?.[1]),text:segment};
    });
}

export function matchesAttributedEvidence(claim,sourceText) {
    if(!claim?.evidence || !claim?.sourceUrl) return false;
    const evidence=normalizeWords(claim.evidence);
    // Extremely short quotes cannot prove an independent factual claim.
    if(evidence.length<18 || evidence.split(" ").length<4) return false;
    const sections=sourceSections(sourceText);
    if(sections){
        const url=canonical(claim.sourceUrl);
        const section=sections.find(s=>s.url===url);
        if(!url||!section) return false;
        return normalizeWords(section.text).includes(evidence);
    }
    // Legacy single-source validation may provide no section headers.
    return normalizeWords(sourceText).includes(evidence);
}
