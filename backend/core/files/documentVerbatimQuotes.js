import {createHash} from "node:crypto";

const hash = text => createHash("sha256").update(text, "utf8").digest("hex");

/**
 * Exact quote offsets are stricter than normalized quotation validation.
 * Ambiguous or typographically normalized quotes never get invented offsets.
 */
export function locateVerbatimQuoteRanges(window, bundle, checked) {
    if (!window || typeof window.text !== "string" ||
        !Number.isSafeInteger(window.number) || window.number < 1 ||
        !Number.isSafeInteger(window.sectionIndex) || window.sectionIndex < 0 ||
        !Number.isSafeInteger(window.startOffset) || window.startOffset < 0 ||
        window.endOffset !== window.startOffset + window.text.length ||
        !bundle?.exactSourceCoverage || !(bundle.index instanceof Map) ||
        checked?.success !== true || !Array.isArray(checked.verifiedFindings) ||
        !checked.verifiedFindings.length) return null;

    const validRefs = new Set(checked.citedRefs);
    const ranges = [];
    const dedup = new Set();
    let evidenceQuotes = 0, exactLocatedQuotes = 0;
    let ambiguousQuotes = 0, nonLiteralQuotes = 0;
    for (const finding of checked.verifiedFindings) {
        if (!Array.isArray(finding?.evidence) || !finding.evidence.length ||
            finding.evidence.length > 3) return null;
        for (const item of finding.evidence) {
            evidenceQuotes++;
            const quote = item?.quote;
            const fragment = bundle.index.get(item?.ref);
            if (typeof quote !== "string" || !quote.trim() || quote.length > 260 ||
                !validRefs.has(item.ref) || !fragment ||
                !Number.isSafeInteger(fragment.startOffset) ||
                !Number.isSafeInteger(fragment.endOffset) ||
                fragment.startOffset < 0 ||
                fragment.endOffset > window.text.length ||
                window.text.slice(fragment.startOffset, fragment.endOffset) !== fragment.text)
                return null;
            const local = fragment.text.indexOf(quote);
            if (local === -1) {
                nonLiteralQuotes++;
                continue;
            }
            if (fragment.text.indexOf(quote, local + 1) !== -1) {
                ambiguousQuotes++;
                continue;
            }
            const startOffset = window.startOffset + fragment.startOffset + local;
            const endOffset = startOffset + quote.length;
            if (window.text.slice(startOffset - window.startOffset,
                endOffset - window.startOffset) !== quote) return null;
            exactLocatedQuotes++;
            const ref = `О${window.number}:${item.ref}`;
            const key = ref + ":" + startOffset + ":" + endOffset;
            if (!dedup.has(key)) {
                dedup.add(key);
                ranges.push({ref, windowNumber:window.number,
                    sectionIndex:window.sectionIndex,
                    startOffset,endOffset,quoteSha256:hash(quote)});
            }
        }
    }
    return {ranges,evidenceQuotes,exactLocatedQuotes,ambiguousQuotes,
        nonLiteralQuotes,
        allVerbatimQuoteOffsetsLocated: evidenceQuotes>0 &&
            exactLocatedQuotes===evidenceQuotes};
}

export function scopeVerbatimQuoteRanges(ranges, fileIndex) {
    if (!Array.isArray(ranges) ||
        !Number.isSafeInteger(fileIndex) || fileIndex < 1 || fileIndex > 3)
        return null;
    const result=[];
    for (const row of ranges) {
        if (typeof row?.ref !== "string" ||
            !/^О\d+:Ф1:Ф\d+$/u.test(row.ref) ||
            !Number.isSafeInteger(row.startOffset) ||
            !Number.isSafeInteger(row.endOffset) ||
            row.endOffset <= row.startOffset ||
            !/^[a-f0-9]{64}$/u.test(String(row.quoteSha256||"")))
            return null;
        result.push({...row,fileIndex,ref:`Д${fileIndex}:${row.ref}`});
    }
    return result;
}
