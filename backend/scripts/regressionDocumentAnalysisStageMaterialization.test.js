import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages}
 from "../core/files/documentAnalysisStagePlan.js";
import {materializeVerifiedDocumentAnalysisStage}
 from "../core/files/documentAnalysisStageMaterialization.js";

const sha=s=>createHash("sha256").update(s,"utf8").digest("hex");
const source="ВВЕДЕНИЕ\n"+"Норма права и судебная практика. ".repeat(19500)+
    "\nЗАКЛЮЧЕНИЕ\nОсновные выводы.";
function fixture(){
    const hash=sha(source);
    const first=materializeDocumentOverflowBatch(source,hash,1);
    assert.ok(first);
    const batches=[];
    for(let i=1;i<=first.batchCount;i++){
        const part=materializeDocumentOverflowBatch(source,hash,i);
        assert.ok(part);batches.push(part);
    }
    const verified=verifyMaterializedOverflowBatches(source,hash,batches);
    assert.ok(verified?.complete);
    assert.ok(verified.totalWindows>32);
    return verified;
}
test("every stage reassembles to exact original source without omissions",()=>{
    const verified=fixture(),plan=planVerifiedDocumentAnalysisStages(source,verified);
    assert.ok(plan);
    let cursor=0,number=1;
    const texts=[];
    for(let index=1;index<=plan.stageCount;index++){
        const stage=materializeVerifiedDocumentAnalysisStage(source,verified,index);
        assert.ok(stage);
        assert.equal(stage.stageNumber,index);
        assert.equal(stage.stageCount,plan.stageCount);
        assert.equal(stage.stageSha256,plan.stages[index-1].stageSha256);
        assert.equal(stage.startOffset,cursor);
        assert.equal(stage.firstWindowNumber,number);
        assert.ok(stage.windowCount<=16);
        assert.ok(stage.textBytes<=650000);
        assert.equal(stage.sourceOffsetsVerified,true);
        assert.equal(stage.stageIntegrityVerified,true);
        assert.equal(stage.modelAnalysisPerformed,false);
        assert.equal(stage.stageAnalysisComplete,false);
        assert.equal(stage.fullSourceAnalysisConfirmed,false);
        for(const window of stage.windows){
            assert.equal(window.number,number++);
            assert.equal(window.startOffset,cursor);
            assert.equal(window.windowSha256,sha(window.text));
            assert.equal(window.text,source.slice(window.startOffset,window.endOffset));
            texts.push(window.text);
            cursor=window.endOffset;
        }
        assert.equal(stage.endOffset,cursor);
        assert.equal(stage.nextStageNumber,
            index===plan.stageCount?null:index+1);
    }
    assert.equal(cursor,source.length);
    assert.equal(number,verified.totalWindows+1);
    assert.equal(texts.join(""),source);
});
test("rejects source, hash, stage index or window tampering",()=>{
    const verified=fixture();
    const f=(s,v,n)=>materializeVerifiedDocumentAnalysisStage(s,v,n);
    assert.equal(f(source,verified,0),null);
    assert.equal(f(source,verified,-1),null);
    assert.equal(f(source,verified,1.5),null);
    assert.equal(f(source,verified,999),null);
    assert.equal(f(source+"Z",verified,1),null);
    assert.equal(f(source,{...verified,sourceSha256:"0".repeat(64)},1),null);
    assert.equal(f(source,{...verified,analyzedWindows:1},1),null);
    assert.equal(f(source,{...verified,windows:verified.windows.map((w,i)=>
        i===2?{...w,text:"Z"+w.text.slice(1)}:w)},1),null);
    assert.equal(f(source,{...verified,windows:verified.windows.slice(1)},1),null);
});
test("stage text limit guards response size and never fabricates analysis",()=>{
    const chinese="ВВЕДЕНИЕ\n"+"汉".repeat(580000);
    const hash=sha(chinese);
    const first=materializeDocumentOverflowBatch(chinese,hash,1);
    assert.ok(first);
    const batches=[];
    for(let i=1;i<=first.batchCount;i++){
        const batch=materializeDocumentOverflowBatch(chinese,hash,i);
        assert.ok(batch);batches.push(batch);
    }
    const verified=verifyMaterializedOverflowBatches(chinese,hash,batches);
    assert.ok(verified.totalWindows>32);
    assert.equal(materializeVerifiedDocumentAnalysisStage(chinese,verified,1),null);
});
