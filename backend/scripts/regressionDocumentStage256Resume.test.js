import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {performance} from "node:perf_hooks";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence}
 from "../core/files/documentWindowEvidence.js";
import {materializeVerifiedDocumentOverflowBatches}
 from "../core/files/documentOverflowBatchBulkMaterialization.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {verifyDocumentStageReceipt}
 from "../core/files/documentStageReceiptAuth.js";
import {makeDocumentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";
import {getAIProviderAvailability}
 from "../ai/providerAvailability.js";

const source=makeDocumentStressSource();
const sha=hashDocumentStressSource(source);
const digest=x=>createHash("sha256").update(x,"utf8").digest("hex");
const bulk=materializeVerifiedDocumentOverflowBatches(source,sha);
if(!bulk?.verified?.complete||bulk.verified.totalWindows!==256||
   bulk.batches.length!==16)throw Error("Fixture must have 256 validated windows");
const pages=[];
for(let offset=0;offset!==null;){
    const page=pageDocumentText(source,offset);
    pages.push({...page,name:"huge.docx"});
    offset=page.nextOffset;
}
const task="Исследуй 256 окон с проверенными цитатами";
const base={name:"huge.docx",pages,overflowBatches:bulk.batches};
const input=item=>({task,attachments:[{...base,...item}]});
async function syntheticModel(_prompt,_task,attachments){
    const fragment=createLosslessWindowEvidence(attachments[0])?.fragments[0];
    assert.ok(fragment?.text.length>=6);
    const quote=fragment.text.split(/\r?\n/u).find(line=>line.trim().length>=6)?.trim().slice(0,32);
    assert.ok(quote.length>=6);
    return JSON.stringify({sections:[{heading:"Основания",findings:[{
        kind:"author_claim",text:"Подтверждённый фрагмент исходного документа",
        evidence:[{ref:fragment.ref,quote}]
    }]}]});
}
function quota(){
    const error=new Error("requests per minute exceeded");
    error.status=429;
    error.headers={"retry-after":"120"};
    return error;
}
async function keyed(t,run){
    const old=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    process.env.JESSICA_STAGE_ATTESTATION_KEY="c".repeat(64);
    let ms=Date.now()+5_000_000;
    t.mock.method(Date,"now",()=>ms);
    t.mock.method(console,"warn",()=>{});
    try{return await run({advance:delta=>{ms+=delta;},
        key:"c".repeat(64)});}
    finally{
        if(old===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
        else process.env.JESSICA_STAGE_ATTESTATION_KEY=old;
    }
}

test("256 windows: signed stages 1-15, 429 at window 243, resume and final merge",
 async t=>keyed(t,async({advance,key})=>{
    const start=performance.now();
    const reports=[];
    let normalCalls=0;
    for(let stageNumber=1;stageNumber<=15;stageNumber++){
        const result=await analyzeDocumentAttachments(input({
            stageNumber,
            ...(stageNumber>1?{stageReports:reports}: {})
        }),async(...args)=>{
            normalCalls++;
            return syntheticModel(...args);
        });
        assert.equal(result.success,true,JSON.stringify({
            stageNumber,resultStage:result.stage,
            text:result.text?.slice(0,180)}));
        assert.equal(result.stage,"document-stage-verified");
        assert.equal(result.readingCoverage.processedWindows,16);
        assert.equal(verifyDocumentStageReceipt(result,key),true);
        reports.push(result);
    }
    assert.equal(normalCalls,240);
    let attempts=0;
    const interrupted=await analyzeDocumentAttachments(input({
        stageNumber:16,stageReports:reports
    }),async(...args)=>{
        if(++attempts===3)throw quota();
        return syntheticModel(...args);
    });
    assert.equal(interrupted.success,false);
    assert.equal(interrupted.stage,"ai-provider-cooldown");
    assert.equal(attempts,3);
    const coverage=interrupted.readingCoverage;
    assert.equal(coverage.totalSourceWindows,256);
    assert.equal(coverage.stageNumber,16);
    assert.equal(coverage.processedWindows,2);
    assert.equal(coverage.interruptedWindow,243);
    assert.equal(coverage.resumeAvailable,true);
    assert.equal(coverage.fullSourceAnalysisConfirmed,false);
    const checkpoint=coverage.resumeCheckpoint;
    assert.equal(checkpoint.windows.length,2);
    assert.deepEqual(checkpoint.windows.map(x=>x.number),[241,242]);
    assert.equal(checkpoint.sourceSha256,sha);
    const noCall=async()=>{throw Error("Model must not be called");};
    const during=await analyzeDocumentAttachments(input({
        stageNumber:16,stageReports:reports,resumeCheckpoint:checkpoint
    }),noCall);
    assert.equal(during.stage,"ai-provider-cooldown");
    assert.equal(getAIProviderAvailability().blocked,true);
    advance(120000);
    assert.equal(getAIProviderAvailability().available,true);
    let rejectedCalls=0;
    const fake=structuredClone(checkpoint);
    fake.rawResponses[0]=fake.rawResponses[0].replace(
        "Подтверждённый","Поддельный");
    const bad=await analyzeDocumentAttachments(input({
        stageNumber:16,stageReports:reports,resumeCheckpoint:fake
    }),async()=>{rejectedCalls++;throw Error("Forged prefix executed");});
    assert.equal(bad.success,false);
    assert.equal(bad.stage,"document-stage-resume-invalid");
    assert.equal(rejectedCalls,0);

    let remaining=0;
    const finalStage=await analyzeDocumentAttachments(input({
        stageNumber:16,stageReports:reports,resumeCheckpoint:checkpoint
    }),async(...args)=>{
        remaining++;
        return syntheticModel(...args);
    });
    assert.equal(finalStage.success,true,finalStage.stage);
    assert.equal(remaining,14);
    assert.equal(finalStage.readingCoverage.processedWindows,16);
    assert.equal(finalStage.readingCoverage.verifiedWindows.length,16);
    assert.equal(finalStage.readingCoverage.verifiedWindows[0].windowNumber,241);
    assert.equal(finalStage.readingCoverage.verifiedWindows.at(-1).windowNumber,256);
    assert.equal(verifyDocumentStageReceipt(finalStage,key),true);
    assert.equal(finalStage.continuation.awaitFinalMerge,true);
    const merged=await analyzeDocumentAttachments(input({
        stageReports:[...reports,finalStage]
    }),noCall);
    assert.equal(merged.success,true,JSON.stringify({
        stage:merged.stage,text:merged.text?.slice(0,140)}));
    assert.equal(merged.stage,"document-stages-authenticated");
    assert.equal(merged.readingCoverage.processedWindows,256);
    assert.equal(merged.readingCoverage.authenticatedStages,16);
    assert.equal(merged.readingCoverage.reportBodySourceQuotesChecked,true);
    assert.equal(merged.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(merged.readingCoverage.claimTruthVerified,false);
    process.stdout.write("DOCUMENT_256_STAGE_RESUME "+
        JSON.stringify({windows:256,authenticatedStages:16,
            verifiedBefore429:242,interruptedWindow:243,
            lastStageReused:2,lastStageModelCalls:14,
            totalModelCalls:normalCalls+attempts+remaining,
            fullySourceAudited:true,semanticTruthVerified:false,
            durationMs:Math.round(performance.now()-start)})+"\n");
 }));

test("256-window stage rejects wrong source and task without model invocation",
 async t=>keyed(t,async({key})=>{
    const stage=await analyzeDocumentAttachments(input({stageNumber:1}),
        syntheticModel);
    assert.equal(stage.success,true,stage.stage);
    assert.equal(verifyDocumentStageReceipt(stage,key),true);
    let calls=0;
    const changedTask=await analyzeDocumentAttachments({
        task:task+" (changed)",attachments:[{...base,stageNumber:2,
            stageReports:[stage]}]
    },async()=>{calls++;throw Error("Unverified task");});
    assert.equal(changedTask.success,false);
    assert.equal(changedTask.stage,"document-stage-resume-invalid");
    const altered=structuredClone(base);
    altered.pages[0].text="Подмена"+altered.pages[0].text.slice(7);
    const changedSource=await analyzeDocumentAttachments(input({
        ...altered,stageNumber:2,stageReports:[stage]
    }),async()=>{calls++;throw Error("Changed source");});
    assert.equal(changedSource.success,false);
    assert.equal(changedSource.stage,"document-overflow-batches-invalid");
    assert.equal(calls,0);
 }));
