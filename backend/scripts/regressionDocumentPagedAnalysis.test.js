import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createDocumentEvidence} from "../core/files/documentEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";

const buildPages=(text,name="scientific.docx")=>{
 const pages=[];let offset=0;
 do{const page=pageDocumentText(text,offset);
    pages.push({...page,name});offset=page.nextOffset;
 }while(offset!==null);
 return pages;
};
const source="ВВЕДЕНИЕ\n"+"Научная проблема требует анализа. ".repeat(900)+
 "\nГЛАВА 1. ТЕОРЕТИЧЕСКИЕ ОСНОВАНИЯ\n"+
 "Правовая позиция рассматривается подробно. ".repeat(900)+
 "\n1.1 ПРАКТИЧЕСКИЕ ВОПРОСЫ\n"+
 "Техническая документация содержит данные. ".repeat(700)+
 "\nЗАКЛЮЧЕНИЕ\nРезультаты представлены.";
const answer=async(_,__,files)=>{
 const item=createDocumentEvidence(files).fragments[0];
 return JSON.stringify({sections:[{heading:"Анализ",findings:[{
    text:"Автор приводит проверяемое утверждение",kind:"author_claim",
    evidence:[{ref:item.ref,quote:item.text.slice(0,35)}]}]}]});
};
test("long document: all source windows reach the model with scoped citations",async()=>{
 const plan=planDocumentSectionWindows(source);
 assert.equal(plan.complete,true);
 assert.ok(plan.windows.length>=5);
 assert.equal(plan.windows.map(p=>p.text).join(""),source);
 const seen=[];
 const result=await analyzeDocumentAttachments({task:"Проанализируй положения",
  attachments:[{name:"scientific.docx",pages:buildPages(source)}]},
  async(prompt,task,files)=>{seen.push(files[0].text);return answer(prompt,task,files);});
 assert.equal(result.success,true);
 assert.equal(seen.join(""),source);
 assert.equal(result.readingCoverage.processedWindows,plan.windows.length);
 assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
 assert.match(result.text,/ЗАКЛЮЧЕНИЕ/);
 assert.match(result.text,/\[О2:Ф1:Ф1\]/);
});
test("missing or altered source pages never reach model",async()=>{
 const pages=buildPages(source);
 for(const invalid of [pages.slice(1),pages.slice(0,-1),
   [{...pages[0],text:"X"+pages[0].text.slice(1)},...pages.slice(1)]]){
   let count=0;
   const got=await analyzeDocumentAttachments({task:"Проверь",
     attachments:[{name:"scientific.docx",pages:invalid}]},
     async()=>{count++;return "{}";});
   assert.equal(got.success,false);
   assert.equal(got.stage,"document-pages");
   assert.equal(count,0);
 }
});
test("overlong document explicitly requires continuation rather than false success",async()=>{
 const large="ГЛАВА 1. Материал\n"+"Правовое положение. ".repeat(17000);
 const plan=planDocumentSectionWindows(large);
 assert.equal(plan.complete,false);
 assert.equal(plan.reason,"window-limit");
 let called=0;
 const result=await analyzeDocumentAttachments({task:"Проанализируй",
     attachments:[{name:"scientific.docx",pages:buildPages(large)}]},
     async()=>{called++;return "{}";});
 assert.equal(result.stage,"document-read-plan");
 assert.equal(result.readingCoverage.complete,false);
 assert.equal(called,0);
});
test("one invalid window stops success and accurately reports processing",async()=>{
 let calls=0;
 const plan=planDocumentSectionWindows(source);
 const result=await analyzeDocumentAttachments({task:"Проверь",
  attachments:[{name:"scientific.docx",pages:buildPages(source)}]},
  async(prompt,task,files)=>{calls++;return calls===3?'{"sections":[]}':answer(prompt,task,files);});
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-window-evidence");
 assert.equal(result.readingCoverage.processedWindows,2);
 assert.equal(result.readingCoverage.plannedWindows,plan.windows.length);
});
