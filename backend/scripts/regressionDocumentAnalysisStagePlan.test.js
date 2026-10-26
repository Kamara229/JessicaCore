import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {materializeDocumentOverflowBatch}
    from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
    from "../core/files/documentOverflowBatchVerification.js";
import {planVerifiedDocumentAnalysisStages}
    from "../core/files/documentAnalysisStagePlan.js";
import {analyzeDocumentAttachments}
    from "../core/files/documentAnalysisFacade.js";

const sha=text=>createHash("sha256").update(text,"utf8").digest("hex");
const source="ВВЕДЕНИЕ\n"+"Правовая оценка спорного вопроса. ".repeat(19000)+
    "\nЗАКЛЮЧЕНИЕ\nИтоги анализа.";
const pageList=()=>{
    let at=0;const pages=[];
    do{const page=pageDocumentText(source,at);
       assert.ok(page?.text);
       pages.push({...page,name:"thesis.docx"});
       at=page.nextOffset;
    }while(at!==null);
    return pages;
};
const make=()=>{
    const hash=sha(source);
    const first=materializeDocumentOverflowBatch(source,hash,1);
    assert.ok(first);
    const batches=[];
    for(let n=1;n<=first.batchCount;n++){
        const item=materializeDocumentOverflowBatch(source,hash,n);
        assert.ok(item);batches.push(item);
    }
    const verified=verifyMaterializedOverflowBatches(source,hash,batches);
    assert.ok(verified?.complete);
    assert.ok(verified.totalWindows>32);
    return {hash,batches,verified};
};

test("33+ source windows become contiguous SHA-256 stage manifest",()=>{
    const {verified}=make();
    const plan=planVerifiedDocumentAnalysisStages(source,verified);
    assert.ok(plan);
    assert.equal(plan.stageCount,Math.ceil(verified.totalWindows/16));
    assert.equal(plan.stageCount,plan.stages.length);
    assert.equal(plan.processedWindows,0);
    assert.equal(plan.completedStages,0);
    assert.equal(plan.modelAnalysisPerformed,false);
    assert.equal(plan.resumeImplemented,false);
    assert.equal(plan.semanticCompletenessConfirmed,false);
    assert.equal(plan.nextStageNumber,1);
    let next=0,windowNumber=1;
    for(const stage of plan.stages){
        assert.equal(stage.stageNumber,plan.stages.indexOf(stage)+1);
        assert.ok(stage.windowCount<=16);
        assert.equal(stage.startOffset,next);
        assert.equal(stage.firstWindowNumber,windowNumber);
        assert.equal(stage.stageSha256,sha(source.slice(stage.startOffset,stage.endOffset)));
        assert.ok(stage.sectionIndexes.length>0);
        next=stage.endOffset;
        windowNumber=stage.lastWindowNumber+1;
        assert.equal(Object.hasOwn(stage,"text"),false);
    }
    assert.equal(next,source.length);
    assert.equal(windowNumber,verified.totalWindows+1);
});

test("stages require cryptographically matching full source and windows",()=>{
    const {verified}=make();
    assert.equal(planVerifiedDocumentAnalysisStages(source+"X",verified),null);
    assert.equal(planVerifiedDocumentAnalysisStages(source,{
        ...verified,sourceSha256:"0".repeat(64)}),null);
    assert.equal(planVerifiedDocumentAnalysisStages(source,{
        ...verified,windows:verified.windows.slice(1)}),null);
    assert.equal(planVerifiedDocumentAnalysisStages(source,{
        ...verified,windows:verified.windows.map((w,i)=>i===3?
          {...w,text:"X"+w.text.slice(1)}:w)}),null);
    assert.equal(planVerifiedDocumentAnalysisStages(source,{
        ...verified,analyzedWindows:1}),null);
    assert.equal(planVerifiedDocumentAnalysisStages(source,{
        ...verified,complete:false}),null);
});

test("long document reports stage plan but does not call model or claim success",async()=>{
    const {batches,verified}=make();
    let calls=0;
    const response=await analyzeDocumentAttachments({
        task:"Анализ всех глав диссертации",
        attachments:[{name:"thesis.docx",pages:pageList(),
           overflowBatches:batches}]
    },async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(response.success,false);
    assert.equal(response.stage,"document-overflow-analysis-pending");
    assert.equal(response.readingCoverage.processedWindows,0);
    assert.equal(response.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(response.readingCoverage.stagePlan.totalWindows,verified.totalWindows);
    assert.equal(response.readingCoverage.stagePlan.modelAnalysisPerformed,false);
    assert.equal(response.readingCoverage.stagePlan.nextStageNumber,1);
    assert.equal(JSON.stringify(response.readingCoverage.stagePlan).includes(
       "Правовая оценка"),false);
});

test("partial batch series produces no stage plan and no model calls",async()=>{
    const {batches}=make();let calls=0;
    const result=await analyzeDocumentAttachments({
        task:"Анализ диссертации",
        attachments:[{name:"thesis.docx",pages:pageList(),
            overflowBatches:batches.slice(0,1)}]
    },async()=>{calls++;return "{}";});
    assert.equal(calls,0);
    assert.equal(result.stage,"document-overflow-batches-incomplete");
    assert.equal(result.readingCoverage.stagePlan,undefined);
});
