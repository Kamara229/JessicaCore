import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";

const sha=s=>createHash("sha256").update(s,"utf8").digest("hex");
const large="ВВЕДЕНИЕ\n"+("Аргументация гражданского права. ".repeat(12000))+
    "\nГЛАВА 1. Правовая природа\n"+
    ("Научная позиция и судебная практика. ".repeat(1800))+
    "\nЗАКЛЮЧЕНИЕ\nРезультаты исследования.";
const materialize=()=>{
    const first=materializeDocumentOverflowBatch(large,sha(large),1);
    assert.ok(first);
    const batches=[];
    for(let i=1;i<=first.batchCount;i++){
        const next=materializeDocumentOverflowBatch(large,sha(large),i);
        assert.ok(next);
        batches.push(next);
    }
    return batches;
};
const pages=s=>{
    const all=[];let offset=0;
    do{const part=pageDocumentText(s,offset);all.push({...part,name:"paper.docx"});
       offset=part.nextOffset;}while(offset!==null);
    return all;
};
test("the full original text is partitioned exactly across trusted batches",()=>{
    const batches=materialize(),out=verifyMaterializedOverflowBatches(
        large,sha(large),batches);
    assert.ok(out?.complete);
    assert.equal(out.allSourceWindowsVerified,true);
    assert.equal(out.collectedChars,large.length);
    assert.equal(out.verifiedBatches,batches.length);
    assert.equal(out.verifiedWindows,out.totalWindows);
    assert.equal(out.windows.map(w=>w.text).join(""),large);
    assert.equal(out.modelAnalysisPerformed,false);
    assert.equal(out.analyzedWindows,0);
    assert.equal(out.semanticCompletenessConfirmed,false);
});
test("incomplete prefix is explicit and offers no windows to the AI",()=>{
    const batches=materialize();
    assert.ok(batches.length>=2);
    const partial=verifyMaterializedOverflowBatches(large,sha(large),
        batches.slice(0,1));
    assert.equal(partial.complete,false);
    assert.equal(partial.allSourceWindowsVerified,false);
    assert.equal(partial.nextBatchNumber,2);
    assert.equal(Object.hasOwn(partial,"windows"),false);
});
test("reordered, repeated, forged, cross-source batches all fail closed",()=>{
    const full=materialize(),first=full[0],last=full.at(-1);
    for(const bad of [
        [last,first],[first,first],
        [first,{...last,startOffset:last.startOffset+1}],
        [first,{...last,sourceSha256:"0".repeat(64)}],
        [first,{...last,modelAnalysisPerformed:true}],
        [first,{...last,analyzedWindows:1}],
        [first,{...last,windows:last.windows.map((w,i)=>
           i===0?{...w,text:"Х"+w.text.slice(1)}:w)}]
    ])assert.equal(verifyMaterializedOverflowBatches(large,sha(large),bad),null);
    assert.equal(verifyMaterializedOverflowBatches(large+"Z",sha(large),full),null);
    assert.equal(verifyMaterializedOverflowBatches(large,sha(large),[]),null);
});
test("facade rejects incomplete source and unsupported model evidence",async()=>{
    const batches=materialize();
    let calls=0;
    for(const [series,stage] of [
        [batches.slice(0,1),"document-overflow-batches-incomplete"],
        [batches,"document-window-evidence"]
    ]){
        const result=await analyzeDocumentAttachments({task:"Изучить все главы",
            attachments:[{name:"paper.docx",pages:pages(large),
                overflowBatches:series}]},async()=>{calls++;return "{}";});
        assert.equal(result.success,false);
        assert.equal(result.stage,stage);
        assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
        assert.equal(result.readingCoverage.processedWindows,0);
    }
    assert.equal(calls,1);
});
