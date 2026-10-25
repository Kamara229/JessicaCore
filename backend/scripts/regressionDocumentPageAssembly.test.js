import test from "node:test";
import assert from "node:assert/strict";
import { pageDocumentText } from "../core/files/documentTextWindow.js";
import { assembleDocumentPages } from "../core/files/documentPageAssembly.js";
import { normalizeAttachmentContexts } from "../core/files/attachmentContext.js";

const buildPages = text => {
    const pages = [];
    let offset = 0;
    do {
        const page = pageDocumentText(text, offset);
        pages.push({ ...page, name: "dissertation.docx" });
        offset = page.nextOffset;
    } while (offset !== null);
    return pages;
};

test("full long-form document is reconstructed byte-for-byte across pages", () => {
    const input = "Глава 1.\n" + "Авторское положение. ".repeat(3500) + "🧪";
    const pages = buildPages(input);
    const merged = assembleDocumentPages(pages);
    assert.ok(pages.length > 3);
    assert.equal(merged.text, input);
    assert.equal(merged.complete, true);
    assert.equal(merged.pageCount, pages.length);
});

test("missing, swapped, repeated or overlapping pages are rejected", () => {
    const pages = buildPages("С".repeat(40000));
    for (const invalid of [
        [pages[0], pages[2]],
        [pages[1], pages[0], pages[2]],
        [pages[0], pages[0], pages[2]],
        [pages[0], {...pages[1], offset: 10}, pages[2]],
        [pages[0], {...pages[1], text: "X"+pages[1].text.slice(1)}, pages[2]],
        [pages[0], {...pages[1], name: "different.docx"}, pages[2]],
        [pages[0], {...pages[1], extractedChars: 123}, pages[2]]
    ]) assert.equal(assembleDocumentPages(invalid), null);
});

test("a partial page series cannot masquerade as a complete document", () => {
    const pages = buildPages("д".repeat(45000));
    assert.equal(assembleDocumentPages(pages.slice(0, -1)), null);
    assert.equal(assembleDocumentPages([{...pages[0], hasMore:false, nextOffset:null}]),null);
    assert.equal(assembleDocumentPages([{...pages[0], extractedChars:8}]),null);
});

test("a supplied page series integrates with attachment normalization but context stays bounded", () => {
    const text = "Заголовок. " + "содержимое ".repeat(3800);
    const entry = {name:"dissertation.docx", pages:buildPages(text), truncated:true};
    const merged = assembleDocumentPages(entry.pages);
    assert.equal(merged.text, text);
    const docs = normalizeAttachmentContexts([entry]);
    assert.equal(docs.length, 1);
    assert.equal(docs[0].name,"dissertation.docx");
    assert.ok(text.startsWith(docs[0].text));
    assert.equal(docs[0].truncated, true); // analysis cannot claim full coverage yet
    assert.equal(docs[0].text.length,16000);
    assert.equal(normalizeAttachmentContexts([{...entry,pages:entry.pages.slice(1)}]).length,0);
});

test("short documents and malformed paging metadata", () => {
    const pages = buildPages("Глава 1");
    assert.equal(assembleDocumentPages(pages).text,"Глава 1");
    assert.equal(normalizeAttachmentContexts([{name:"legacy.txt",text:"Текст"}])[0].text,"Текст");
    assert.equal(assembleDocumentPages([]),null);
    assert.equal(assembleDocumentPages([{...pages[0],offset:-1}]),null);
    assert.equal(assembleDocumentPages([{...pages[0],sourceSha256:"x"}]),null);
});
