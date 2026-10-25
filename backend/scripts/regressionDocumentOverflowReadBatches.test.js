import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";

const sha=s=>createHash("sha256").update(s,"utf8").digest("hex");
const pages=source=>{
 const all=[];let offset=0;
 do{const item=pageDocumentText(source,offset);all.push({...item,name:"thesis.docx"});
    offset=item.nextOffset;}while(offset!==null);
 return all;
};
const large="ВВЕДЕНИЕ\n"+"Научное положение и обоснование. ".repeat(9500)+
    "\nГЛАВА 1. ТЕОРИЯ\n"+"Правовая позиция автора. ".repeat(2500)+
    "\nЗАКЛЮЧЕНИЕ\nОкончание.";

test("more than 16 windows get cryptographically checked lossless 16-window batches",()=>{
 const ordinary=planDocumentSectionWindows(large);
 assert.equal(ordinary.complete,false);
 assert.equal(ordinary.reason,"window-limit");
 const plan=planDocumentOverflowBatches(large,sha(large));
 assert.ok(plan);
 assert.ok(plan.totalWindows>16);
 assert.ok(plan.batchCount>=2);
 assert.equal(plan.completeSourcePartition,true);
 assert.equal(plan.analyzedWindows,0);
 assert.equal(plan.modelAnalysisPerformed,false);
 assert.equal(plan.readCompleted,false);
 assert.equal(plan.resumeImplemented,false);
 let cursor=0,n=0;
 for(const batch of plan.batches){
    assert.ok(batch.windowCount>=1 && batch.windowCount<=16);
    assert.equal(batch.startOffset,cursor);
    for(const window of batch.windows){
       assert.equal(window.number,++n);
       assert.equal(window.startOffset,cursor);
       assert.equal(sha(large.slice(window.startOffset,window.endOffset)),window.textSha256);
       assert.equal(Object.hasOwn(window,"text"),false);
       cursor=window.endOffset;
    }
    assert.equal(batch.endOffset,cursor);
    assert.equal(batch.windowCount,batch.windows.length);
 }
 assert.equal(cursor,large.length);
 assert.equal(n,plan.totalWindows);
 assert.equal(plan.batchCount,Math.ceil(n/16));
});

test("more than 16 windows still fail closed before any AI call with honest blueprint",async()=>{
 let calls=0;
 const result=await analyzeDocumentAttachments({
    task:"Прочитай весь документ",attachments:[{name:"thesis.docx",pages:pages(large)}]
 },async()=>{calls++;return "{}";});
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-read-plan");
 assert.equal(calls,0);
 assert.equal(result.readingCoverage.processedWindows,0);
 assert.equal(result.readingCoverage.complete,false);
 assert.equal(result.readingCoverage.reason,"window-limit");
 const plan=result.readingCoverage.overflowBatches;
 assert.ok(plan);
 assert.ok(plan.totalWindows>16);
 assert.equal(plan.sourceSha256,sha(large));
 assert.equal(plan.resumeImplemented,false);
 assert.equal(plan.modelAnalysisPerformed,false);
});

test("tampered input, incorrect checksum, oversized structure and surrogate boundaries fail safe",()=>{
 assert.equal(planDocumentOverflowBatches(large,"0".repeat(64)),null);
 assert.equal(planDocumentOverflowBatches("",sha("")),null);
 const invalid="ГЛАВА 1. ТЕКСТ\n"+("1.1 Тема\nАбзац\n".repeat(130));
 assert.equal(planDocumentOverflowBatches(invalid,sha(invalid)),null);
 const unicode="ВВЕДЕНИЕ\n"+"А".repeat(15989)+"🧪"+"Б".repeat(40000);
 const plan=planDocumentOverflowBatches(unicode,sha(unicode));
 assert.ok(plan);
 for(const batch of plan.batches)for(const w of batch.windows){
    const text=unicode.slice(w.startOffset,w.endOffset);
    assert.equal(sha(text),w.textSha256);
    assert.ok(!/[\uD800-\uDBFF]$/u.test(text));
    assert.ok(!/^[\uDC00-\uDFFF]/u.test(text));
 }
});