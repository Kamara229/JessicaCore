import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    DOCUMENT_WINDOW_CHARS,
    pageDocumentText,
    parseDocumentTextOffset
} from "../core/files/documentTextWindow.js";

test("old clients still receive first 16000 characters and truncation flag", () => {
    const source = "Т".repeat(20000);
    const page = pageDocumentText(source);
    assert.equal(DOCUMENT_WINDOW_CHARS, 16000);
    assert.equal(page.text, source.slice(0, 16000));
    assert.equal(page.truncated, true);
    assert.equal(page.offset, 0);
    assert.equal(page.nextOffset, 16000);
    assert.equal(page.extractedChars, 20000);
    assert.equal(page.hasMore, true);
});

test("paged extraction can reconstruct long documents without missing characters", () => {
    const source = "Раздел 1.\n" + "важные положения.\n".repeat(2500);
    let offset = 0;
    const pages = [];
    do {
        const page = pageDocumentText(source, offset);
        assert.ok(page);
        pages.push(page.text);
        if (!page.hasMore) break;
        assert.ok(page.nextOffset > offset);
        offset = page.nextOffset;
    } while (pages.length < 10);
    assert.ok(pages.length > 2);
    assert.equal(pages.join(""), source);
    assert.equal(pageDocumentText(source, source.length).hasMore, false);
});

test("emoji at the page boundary is never split into invalid halves", () => {
    const source = "а".repeat(15999) + "🧪" + "б".repeat(100);
    const first = pageDocumentText(source, 0);
    const second = pageDocumentText(source, first.nextOffset);
    assert.equal(first.text.length, 15999);
    assert.equal(first.text + second.text, source);
    assert.equal(pageDocumentText(source, 16000), null);
});

test("offsets reject malformed, unsafe, negative or out-of-range values", () => {
    for (const value of ["-1", "1.5", "Infinity", "01", "1e2", " 3", "9007199254740992"])
        assert.equal(parseDocumentTextOffset(value), null);
    assert.equal(parseDocumentTextOffset(), 0);
    assert.equal(parseDocumentTextOffset("0"), 0);
    assert.equal(parseDocumentTextOffset("16000"), 16000);
    assert.equal(pageDocumentText("текст", 6), null);
    assert.equal(pageDocumentText("текст", -1), null);
    assert.deepEqual(pageDocumentText("текст"), {
        text:"текст", truncated:false, extractedChars:5,
        offset:0, nextOffset:null, hasMore:false,
        sourceSha256: createHash("sha256").update("текст").digest("hex")
    });
});
