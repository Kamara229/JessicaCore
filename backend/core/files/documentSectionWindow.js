import { pageDocumentText, parseDocumentTextOffset } from "./documentTextWindow.js";

/** Safely return a bounded page from one indexed document section. */
export function parseSectionIndex(value) {
    if (value === undefined || value === null || value === "") return null;
    return parseDocumentTextOffset(value);
}

export function selectDocumentSectionPage(source, index, number, offset = null) {
    if (typeof source !== "string" || index?.complete !== true ||
        !Array.isArray(index.sections) ||
        !Number.isSafeInteger(number) || number < 0 ||
        number >= index.sections.length) return null;
    const section = index.sections[number];
    if (!section || !Number.isSafeInteger(section.startOffset) ||
        !Number.isSafeInteger(section.endOffset) ||
        section.endOffset > source.length ||
        section.startOffset < 0 || section.endOffset <= section.startOffset) return null;
    const cursor = offset === null ? section.startOffset : offset;
    if (!Number.isSafeInteger(cursor) || cursor < section.startOffset ||
        cursor >= section.endOffset) return null;
    const page = pageDocumentText(source, cursor);
    if (!page?.text) return null;
    const size = Math.min(page.text.length, section.endOffset - cursor);
    const text = page.text.slice(0, size);
    const hasMore = cursor + size < section.endOffset;
    return {
        ...page,
        text,
        offset: cursor,
        nextOffset: hasMore ? cursor + size : null,
        hasMore,
        section: {
            index: number,
            title: section.title,
            level: section.level,
            startOffset: section.startOffset,
            endOffset: section.endOffset,
            complete: !hasMore
        }
    };
}
