import test from "node:test";
import assert from "node:assert/strict";
import {buildDocumentSectionIndex} from "../core/files/documentSectionIndex.js";

test("sections after the first 16000 characters remain discoverable",()=>{
    const text="ВВЕДЕНИЕ\n"+ "Предисловие\n".repeat(2000)
        +"ГЛАВА 1. ПРАВОВАЯ ПРИРОДА ЗАЛОГА\n"
        +"1.1 Экономическая природа\n"+"Описание.\n".repeat(15)
        +"1.2 Правовая природа\n"+"ЗАКЛЮЧЕНИЕ\n";
    const index=buildDocumentSectionIndex(text);
    assert.equal(index.complete,true);
    assert.equal(index.sourceChars,text.length);
    assert.ok(index.sections.find(x=>x.title.startsWith("ГЛАВА 1")).startOffset>16000);
    assert.deepEqual(index.sections.map(x=>x.level),[0,1,2,2,0]);
    for(const item of index.sections) {
        assert.equal(text.slice(item.startOffset,item.startOffset+item.title.length),item.title);
        assert.ok(item.endOffset>item.startOffset);
    }
    assert.equal(index.sections.at(-1).endOffset,text.length);
});

test("technical and scientific headings are navigation candidates, not numbered references",()=>{
    const text="Раздел 2. Технические решения\n2.1 Требования к вентиляции\n"
      +"[1, с. 38] источник\n2019. Статья опубликована.\nСПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ";
    const titles=buildDocumentSectionIndex(text).sections.map(x=>x.title);
    assert.equal(titles.length,3);
    assert.match(titles[0],/Раздел 2/);
    assert.match(titles[1],/2.1/);
    assert.match(titles[2],/СПИСОК/);
});

test("unheaded document has deterministic full-span fallback",()=>{
    const text="Обычный абзац без структурных заголовков.";
    assert.deepEqual(buildDocumentSectionIndex(text),{
        sections:[{title:"Документ",level:0,startOffset:0,endOffset:text.length}],
        indexedChars:text.length,sourceChars:text.length,complete:true
    });
    assert.deepEqual(buildDocumentSectionIndex(""),{sections:[],indexedChars:0,sourceChars:0,complete:true});
});

test("large or excessively headed source explicitly reports incomplete indexing",()=>{
    const lines=Array.from({length:150},(_,n)=>`1.${n%50} Заголовок ${n}\n`).join("");
    const over=buildDocumentSectionIndex(lines);
    assert.equal(over.complete,false);
    assert.ok(over.sections.length<=129);
    const gigantic="ГЛАВА 1. Тема\n"+"Содержание\n".repeat(210000);
    const partial=buildDocumentSectionIndex(gigantic);
    assert.equal(partial.complete,false,"too many headings must mark index incomplete");
    assert.ok(partial.sections.length<=129);
    assert.equal(partial.indexedChars,gigantic.length,
        "a heading cap does not truncate a source shorter than scan budget");
    assert.equal(partial.sourceChars,gigantic.length);
    const tooLarge="Без заголовков\\n".replace("\\\\n","\\n").repeat(350000);
    const truncated=buildDocumentSectionIndex(tooLarge);
    assert.equal(truncated.complete,false,"a scan-budget overflow must be explicit");
    assert.ok(truncated.indexedChars<=4_500_000);
    assert.ok(truncated.indexedChars<tooLarge.length);
    assert.equal(truncated.sourceChars,tooLarge.length);
});

test("preface and non-heading lines are covered in ordered offsets",()=>{
    const text="Аннотация текста.\n\nВВЕДЕНИЕ\nОбоснование.\n1.1 Предмет исследования\nОсновной текст.";
    const map=buildDocumentSectionIndex(text);
    assert.equal(map.sections[0].title,"До первого раздела");
    assert.equal(map.sections[0].startOffset,0);
    for(let i=1;i<map.sections.length;i++) {
        assert.equal(map.sections[i-1].endOffset,map.sections[i].startOffset);
    }
});
