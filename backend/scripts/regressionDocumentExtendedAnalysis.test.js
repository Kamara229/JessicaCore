import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {materializeDocumentOverflowBatch} from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches} from "../core/files/documentOverflowBatchVerification.js";
import {buildVerifiedExtendedReadPlan} from "../core/files/documentExtendedReadPlan.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const large="ВВЕДЕНИЕ\n"+"Мотивированное правовое обоснование. ".repeat(9100)+
    "\nГЛАВА 1. Оценка права\n"+
    "Юридический анализ нормативных правил. ".repeat(1250)+
    "\nЗАКЛЮЧЕНИЕ\nИтоговый вывод.";
const digest=async(text)=>{
    const {createHash}=await import("node:crypto");
    return createHash("sha256").update(text,"utf8").digest("hex");
};
const pages=(text,name="thesis.docx")=>{
    const list=[];let at=0;
    do{const p=pageDocumentText(text,at);list.push({...p,name});at=p.nextOffset;}
    while(at!==null);
    return list;
};
const prep=async()=>{
    const sha=await digest(large);
    const first=materializeDocumentOverflowBatch(large,sha,1);
    assert.ok(first);
    const batches=[];
    for(let i=1;i<=first.batchCount;i++){
        batches.push(materializeDocumentOverflowBatch(large,sha,i));
    }
    const checked=verifyMaterializedOverflowBatches(large,sha,batches);
    assert.ok(checked?.complete);
    return {sha,batches,checked};
};
const model=async(_prompt,_task,attached)=>{
    const frag=createLosslessWindowEvidence(attached[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Проверка источника",findings:[{
       kind:"author_claim",text:"Автор излагает обоснование и выводы",
       evidence:[{ref:frag.ref,quote:frag.text.slice(0,Math.min(45,frag.text.length))}]
    }]}]});
};
test("extended verified read plan preserves every source byte and section",async()=>{
    const {sha,checked}=await prep();
    const plan=buildVerifiedExtendedReadPlan(large,sha,checked);
    assert.equal(plan.complete,true);
    assert.ok(plan.windows.length>16);
    assert.ok(plan.windows.length<=32);
    assert.equal(plan.windows.map(w=>w.text).join(""),large);
    assert.equal(plan.expanded,true);
});
test("17-32 verified overflow windows receive source-grounded AI analyses",async()=>{
    const {batches,checked}=await prep();
    const calls=[];
    const result=await analyzeDocumentAttachments({task:"Проверить всю диссертацию",
      attachments:[{name:"thesis.docx",pages:pages(large),overflowBatches:batches}]},
      async(prompt,task,attachments)=>{
         calls.push(attachments[0].text);
         return model(prompt,task,attachments);
      });
    assert.equal(result.success,true,JSON.stringify({stage:result.stage,text:result.text?.slice(0,90)}));
    assert.equal(calls.length,checked.totalWindows);
    assert.equal(calls.join(""),large);
    assert.equal(result.readingCoverage.allWindowsPresented,true);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(result.groundedClaims,checked.totalWindows);
    assert.equal(result.sectionCoverage.allSectionsProcessed,true);
    assert.equal(result.claimProvenance.count,checked.totalWindows);
    assert.equal(result.referenceAudit.sourceChars,large.length);
});
test("incomplete, forged or reordered batch series never triggers AI",async()=>{
    const {sha,batches,checked}=await prep();
    const broken=verified=>buildVerifiedExtendedReadPlan(large,sha,verified);
    assert.equal(broken({...checked,sourceSha256:"0".repeat(64)}),null);
    assert.equal(broken({...checked,windows:checked.windows.slice(1)}),null);
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Проверь полностью",
     attachments:[{name:"thesis.docx",pages:pages(large),
       overflowBatches:batches.slice(0,1)}]},async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(result.success,false);
    assert.equal(result.stage,"document-overflow-batches-incomplete");
});
test("unverified overflow still cannot trigger calls through ordinary page path",async()=>{
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Проверь",
      attachments:[{name:"thesis.docx",pages:pages(large)}]},
      async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(result.stage,"document-read-plan");
});
