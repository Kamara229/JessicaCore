// Literal percentages and currency amounts only. No semantic inference.
const NUMERIC = /(?<![0-9])\d{1,9}(?:[.,]\d{1,3})?[\u00a0\u202f ]?(?:%|₽)(?![0-9])/gu;
const PER_WINDOW = 128;

export function extractExactFactTokens(text, startOffset = 0) {
    if (typeof text !== "string" || !Number.isSafeInteger(startOffset) || startOffset < 0)
        return null;
    const anchors = [];
    let limited = false;
    for (const match of text.matchAll(NUMERIC)) {
        if (anchors.length === PER_WINDOW) { limited = true; break; }
        anchors.push({value: match[0], startOffset: startOffset + match.index,
            endOffset: startOffset + match.index + match[0].length});
    }
    return {anchors, limited};
}
