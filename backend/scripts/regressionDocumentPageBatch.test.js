import test from "node:test";
import assert from "node:assert/strict";
import {buildDocumentPageBatch} from "../core/files/documentPageBatch.js";
import {assembleDocumentPages} from "../core/files/documentPageAssembly.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";

test("single-upload opt-in extracts a complete verified scientific document",()=>{
    const original="ВВЕДЕНИЕ\n"+"Научное содержание. ".repeat(3600)+
        "\nГЛАВА 1\n"+"Важный материал. ".repeat(4200);
    const pages=buildDocumentPageBatch(original,"study.docx");
    assert.ok(pages&&pages.length>=8);
    assert.equal(assembleDocumentPages(pages).text,original);
    assert.equal(pages[0].offset,0);
    assert.equal(pages.at(-1).hasMore,false);
    assert.equal(pages.at(-1).nextOffset,null);
    assert.equal(pages[0].sourceSha256,pages.at(-1).sourceSha256);
    for(const page of pages)assert.equal(page.name,"study.docx");
});

test("emoji boundaries, blank lines and trailing spaces remain exact",()=>{
    const source="🧬".repeat(7800)+"\n\n"+"А".repeat(1100)+"  ";
    const pages=buildDocumentPageBatch(source,"research.txt");
    assert.equal(assembleDocumentPages(pages).text,source);
    assert.equal(pages.map(p=>p.text).join(""),source);
    assert.equal(pageDocumentText(source,0).text,pages[0].text);
});

test("documents requiring over sixteen pages fail closed, no partial batch",()=>{
    const oversized="А".repeat(16*16000+10);
    assert.equal(buildDocumentPageBatch(oversized,"huge.txt"),null);
    assert.equal(buildDocumentPageBatch("","empty.txt"),null);
    assert.equal(buildDocumentPageBatch("Текст",""),null);
});

test("changing page text or order fails verification",()=>{
    const source="Б".repeat(34000);
    const pages=buildDocumentPageBatch(source,"text.txt");
    assert.equal(assembleDocumentPages([pages[1],pages[0],pages[2]]),null);
    assert.equal(assembleDocumentPages([{...pages[0],text:"X"+pages[0].text.slice(1)},...pages.slice(1)]),null);
});
