import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {prepareExtractedDocumentText}
 from "../core/files/documentExtractedTextIntegrity.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {buildDocumentPageBatch} from "../core/files/documentPageBatch.js";
import {assembleDocumentPages} from "../core/files/documentPageAssembly.js";
import {buildDocumentSectionIndex} from "../core/files/documentSectionIndex.js";

const hash=value=>createHash("sha256").update(value,"utf8").digest("hex");

test("leading and trailing document whitespace stays inside the checksum and page offsets",()=>{
    const original=" \n\t  ВВЕДЕНИЕ \r\n\n  Тезис с пробелами.\t\t \n\n";
    const clean=prepareExtractedDocumentText(original);
    assert.equal(clean.text,original);
    assert.equal(clean.hasReadableContent,true);
    assert.equal(clean.nullCharactersRemoved,0);
    const pages=buildDocumentPageBatch(clean.text,"work.docx");
    assert.equal(assembleDocumentPages(pages).text,original);
    assert.equal(pages[0].sourceSha256,hash(original));
    assert.equal(pages[0].extractedChars,original.length);
    const index=buildDocumentSectionIndex(clean.text);
    assert.equal(index.sourceChars,original.length);
    assert.equal(index.sections[0].startOffset,0);
});

test("null characters only are removed, all other line and spacing characters remain",()=>{
    const original="\n\u0000ВВЕДЕНИЕ\nТекст\u0000 \u0000\r\n";
    const result=prepareExtractedDocumentText(original);
    assert.equal(result.text,"\nВВЕДЕНИЕ\nТекст \r\n");
    assert.equal(result.nullCharactersRemoved,3);
    assert.equal(result.hasReadableContent,true);
    assert.equal(pageDocumentText(result.text,0).sourceSha256,hash(result.text));
});

test("whitespace-only and NUL-only documents are rejected without misleading reading success",()=>{
    for(const sample of ["", "\n\t   \r\n", "\u0000\u0000", " \u0000  "]){
        const value=prepareExtractedDocumentText(sample);
        assert.equal(value.hasReadableContent,false);
    }
    assert.equal(prepareExtractedDocumentText(null).hasReadableContent,false);
    assert.equal(prepareExtractedDocumentText(123).hasReadableContent,false);
});

test("emoji and form-feed boundaries preserve extracted text over multiple pages",()=>{
    const raw="\n\f" + "А".repeat(15996) + "🧬" +
      " Продолжение\n" + "Б".repeat(28000) + "\n\n ";
    const clean=prepareExtractedDocumentText(raw);
    const pages=buildDocumentPageBatch(clean.text,"research.pdf");
    assert.ok(pages?.length>=3);
    assert.equal(assembleDocumentPages(pages).text,raw);
    assert.equal(pages.at(-1).extractedChars,raw.length);
    assert.equal(pages[0].sourceSha256,hash(raw));
});
