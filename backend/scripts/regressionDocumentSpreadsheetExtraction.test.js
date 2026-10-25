import test from "node:test";
import assert from "node:assert/strict";
import ExcelJS from "exceljs";
import {
  extractCompleteSpreadsheetText,SpreadsheetExtractionLimitError
} from "../core/files/documentSpreadsheetExtraction.js";
import {prepareExtractedDocumentText}
 from "../core/files/documentExtractedTextIntegrity.js";
import {buildDocumentPageBatch} from "../core/files/documentPageBatch.js";
import {assembleDocumentPages} from "../core/files/documentPageAssembly.js";

const build=async populate=>{
    const book=new ExcelJS.Workbook();
    await populate(book);
    return Buffer.from(await book.xlsx.writeBuffer());
};

test("all worksheets and complex cells are extracted without clipping 120 chars",async()=>{
    const source="В".repeat(180);
    const file=await build(async book=>{
        const one=book.addWorksheet("Главная");
        one.addRow(["Тезис",source]);
        one.addRow(["Формула",{formula:"2+3",result:5}]);
        const two=book.addWorksheet("Приложение");
        two.addRow([{richText:[{text:"Оригинал "},{text:"автора"}]}]);
    });
    const extracted=await extractCompleteSpreadsheetText(file);
    assert.match(extracted,/Лист: Главная/);
    assert.match(extracted,/Лист: Приложение/);
    assert.ok(extracted.includes(source));
    assert.ok(extracted.includes("Оригинал автора"));
    assert.ok(extracted.includes("5"));
    const clean=prepareExtractedDocumentText(extracted);
    assert.equal(clean.hasReadableContent,true);
    const pages=buildDocumentPageBatch(clean.text,"research.xlsx");
    assert.equal(assembleDocumentPages(pages).text,extracted);
});

test("the 4000-cell limit fails closed, never returning only the first rows",async()=>{
    const file=await build(async book=>{
        const sheet=book.addWorksheet("Материалы");
        for(let n=0;n<4001;n++)sheet.addRow(["Материал "+n]);
    });
    await assert.rejects(()=>extractCompleteSpreadsheetText(file),error=>
        error instanceof SpreadsheetExtractionLimitError &&
        error.code==="DOCUMENT_EXTRACTION_INCOMPLETE" &&
        /4000/.test(error.message));
});

test("exactly 4000 nonempty cells are accepted if no later cells exist",async()=>{
    const file=await build(async book=>{
        const sheet=book.addWorksheet("Смета");
        for(let n=0;n<4000;n++)sheet.addRow(["Позиция "+n]);
    });
    const text=await extractCompleteSpreadsheetText(file);
    assert.match(text,/Позиция 3999/);
    assert.match(text,/Позиция 0/);
    assert.equal(text.includes("Позиция 4000"),false);
});

test("oversized single cell is rejected instead of silently cutting it",async()=>{
    const large="А".repeat(256001);
    const file=await build(async book=>{
        const sheet=book.addWorksheet("Большой");
        sheet.addRow([large]);
    });
    await assert.rejects(()=>extractCompleteSpreadsheetText(file),error=>
        error instanceof SpreadsheetExtractionLimitError &&
        /256000/.test(error.message));
});
