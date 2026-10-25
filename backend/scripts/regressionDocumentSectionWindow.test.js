import test from "node:test";
import assert from "node:assert/strict";
import {buildDocumentSectionIndex} from "../core/files/documentSectionIndex.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {parseSectionIndex,selectDocumentSectionPage} from "../core/files/documentSectionWindow.js";

const sample=()=>{
  const source="ВВЕДЕНИЕ\n"+"Вводный текст. ".repeat(2400)+"\n1.1 Теоретические основания\n"+
    "Научное положение. ".repeat(2100)+"\n1.2 Судебная практика\n"+"Факты. ".repeat(2500)+"\nЗАКЛЮЧЕНИЕ\nИтог.";
  return {source,index:buildDocumentSectionIndex(source)};
};
test("retrieve the exact chosen section including sections after 16000 characters",()=>{
  const {source,index}=sample();
  assert.equal(index.complete,true);
  assert.ok(index.sections[1].startOffset>16000);
  let page=selectDocumentSectionPage(source,index,1);
  assert.equal(page.section.title,"1.1 Теоретические основания");
  assert.equal(page.offset,index.sections[1].startOffset);
  assert.equal(page.hasMore,true);
  const snippets=[page.text],hash=page.sourceSha256;
  while(page.hasMore){
    const next=selectDocumentSectionPage(source,index,1,page.nextOffset);
    assert.equal(next.offset,page.nextOffset);
    assert.equal(next.sourceSha256,hash);
    page=next;snippets.push(page.text);
  }
  assert.equal(snippets.join(""),source.slice(index.sections[1].startOffset,index.sections[1].endOffset));
  assert.ok(!snippets.join("").includes("1.2 Судебная практика"));
  assert.equal(page.nextOffset,null);
  assert.equal(page.section.complete,true);
});
test("reject malformed section selectors and offsets outside chosen section",()=>{
  const {source,index}=sample();
  for(const bad of ["-1","01","1.5","NaN","90x"])
    assert.equal(parseSectionIndex(bad),null);
  assert.equal(parseSectionIndex(),null);
  assert.equal(parseSectionIndex("0"),0);
  assert.equal(parseSectionIndex("1"),1);
  assert.equal(selectDocumentSectionPage(source,index,-1),null);
  assert.equal(selectDocumentSectionPage(source,index,300),null);
  assert.equal(selectDocumentSectionPage(source,index,1,index.sections[0].startOffset),null);
  assert.equal(selectDocumentSectionPage(source,index,1,index.sections[1].endOffset),null);
  assert.equal(selectDocumentSectionPage(source,{...index,complete:false},1),null);
  assert.equal(selectDocumentSectionPage(source,index,1,1.5),null);
});
test("existing default page extraction remains unchanged",()=>{
  const {source,index}=sample();
  const regular=pageDocumentText(source);
  const first=selectDocumentSectionPage(source,index,0);
  assert.equal(regular.offset,0);
  assert.equal(regular.text.length,16000);
  assert.equal(first.offset,0);
  assert.ok(first.text.length<=regular.text.length);
  assert.equal(first.sourceSha256,regular.sourceSha256);
});
test("unicode surrogate at a page boundary is never divided",()=>{
  const source="ВВЕДЕНИЕ\n"+"А".repeat(15989)+"🧪"+"Б".repeat(30)+"\n1.1 Глава\nТекст";
  const index=buildDocumentSectionIndex(source);
  const first=selectDocumentSectionPage(source,index,0);
  assert.equal(first.hasMore,true);
  const second=selectDocumentSectionPage(source,index,0,first.nextOffset);
  assert.equal(first.text+second.text,source.slice(0,index.sections[0].endOffset));
  assert.ok(!/[\uD800-\uDBFF]$/u.test(first.text));
});
