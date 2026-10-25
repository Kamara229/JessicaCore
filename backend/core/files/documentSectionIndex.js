/** Deterministic structural navigation for long extracted documents.
 * Offsets refer to the exact normalized text produced by the extractor.
 * A heading candidate is navigation metadata, not proof of semantic structure.
 */
const MAX_SCAN_CHARS = 2_000_000;
const MAX_HEADINGS = 128;
const MAX_TITLE = 180;
const CHAPTER = /^(?:глава|раздел|часть|chapter|section)\s+(?:\d{1,3}|[IVXLC]+)(?:[.\s:—–-]|$)/iu;
const NUMBERED = /^\d{1,2}(?:\.\d{1,2}){1,3}\.?\s+\S/iu;
const SPECIAL = /^(?:введение|заключение|содержание|оглавление|выводы|библиография|список\s+(?:использованных\s+)?(?:источников|литературы)|abstract|references|contents|приложени[ея]\s+[а-яa-z\d]+)$/iu;

function headingLevel(title) {
    if (SPECIAL.test(title)) return 0;
    if (CHAPTER.test(title)) return 1;
    if (NUMBERED.test(title)) {
        const numbered = /^\d{1,2}(?:\.\d{1,2}){1,3}/u.exec(title)?.[0] || "";
        return numbered.split(".").length;
    }
    return null;
}

/** A bounded section map, including preface and explicit incomplete coverage. */
export function buildDocumentSectionIndex(text) {
    if (typeof text !== "string" || !text.length)
        return {sections:[], indexedChars:0, sourceChars:0, complete:true};

    const scanned = Math.min(text.length, MAX_SCAN_CHARS);
    const indexedChars = scanned < text.length
        ? Math.max(0, text.lastIndexOf("\n", scanned - 1) + 1)
        : scanned;
    const headings = [];
    let overflow = false;
    for (const match of text.slice(0, indexedChars).matchAll(/[^\r\n]+/gu)) {
        const title = match[0].trim();
        if (!title || title.length > MAX_TITLE) continue;
        const level = headingLevel(title);
        if (level === null) continue;
        if (headings.length >= MAX_HEADINGS) { overflow = true; break; }
        const indentation = match[0].indexOf(title);
        headings.push({title, level, startOffset:match.index + indentation});
    }
    const complete = !overflow && indexedChars === text.length;
    if (!headings.length) {
        return {sections:[{title:"Документ",level:0,startOffset:0,endOffset:indexedChars}],
            indexedChars, sourceChars:text.length, complete};
    }
    if (headings[0].startOffset > 0)
        headings.unshift({title:"До первого раздела",level:0,startOffset:0});
    const sections = headings.map((heading, i) => ({
        ...heading,
        endOffset: i + 1 < headings.length
            ? headings[i + 1].startOffset : indexedChars
    }));
    return {sections,indexedChars,sourceChars:text.length,complete};
}
