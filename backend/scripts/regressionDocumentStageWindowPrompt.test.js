import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {buildDocumentStageWindowPrompt}
 from "../core/files/documentStageWindowPrompt.js";
import {createLosslessWindowEvidence}
 from "../core/files/documentWindowEvidence.js";

const sha=s=>createHash("sha256").update(s,"utf8").digest("hex");
function record(text){
 return {number:5,title:"Раздел 2.1",sectionIndex:2,startOffset:16000,
    endOffset:16000+text.length,windowSha256:sha(text),text};
}
test("model prompt has every original source fragment in order",()=>{
 const source="Раздел 2.1\n"+"Судебный анализ и проверка данных. ".repeat(380)+
   "\nЭмодзи 🦊 и итоговая позиция.";
 assert.ok(source.length<16000);
 const w=record(source);
 const actual=buildDocumentStageWindowPrompt({task:"Проверь работу",
   name:"paper.docx",window:w,stageNumber:2,stageCount:3});
 assert.ok(actual?.prompt);
 assert.equal(actual.sourceSha256,sha(source));
 assert.equal(actual.sourceChars,source.length);
 assert.deepEqual(actual.excerpt,{name:"paper.docx",text:source,truncated:false});
 const bundle=createLosslessWindowEvidence(actual.excerpt);
 assert.ok(actual.fragments>10);
 assert.ok(actual.prompt.includes(bundle.promptText));
 let cursor=0;
 for(const part of bundle.fragments){
    assert.ok(actual.prompt.indexOf(part.text,cursor)>=cursor);
    cursor=actual.prompt.indexOf(part.text,cursor)+part.text.length;
 }
 assert.equal(bundle.fragments.map(x=>x.text).join(""),source);
 assert.ok(actual.prompt.includes("🦊"));
});
test("tampered, incomplete and invalid windows never reach the model",()=>{
 const original=record("Введение\n"+"Достоверный исходный текст. ".repeat(50));
 const opts={task:"Анализ",name:"paper.docx",window:original,
    stageNumber:1,stageCount:4};
 assert.equal(buildDocumentStageWindowPrompt({...opts,
   window:{...original,text:"Подменённый"+original.text.slice(10)}}),null);
 assert.equal(buildDocumentStageWindowPrompt({...opts,
   window:{...original,endOffset:original.endOffset-1}}),null);
 assert.equal(buildDocumentStageWindowPrompt({...opts,
   window:{...original,text:"Z".repeat(16001),windowSha256:sha("Z".repeat(16001)),
       endOffset:original.startOffset+16001}}),null);
 assert.equal(buildDocumentStageWindowPrompt({...opts,stageNumber:5}),null);
 assert.equal(buildDocumentStageWindowPrompt({...opts,task:"  "}),null);
});
