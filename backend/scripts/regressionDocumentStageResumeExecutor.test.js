import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {verifyDocumentStageReceipt} from "../core/files/documentStageReceiptAuth.js";

const source="ВВЕДЕНИЕ\n"+
    Array.from({length:10000},(_,i)=>
      "Научный аргумент "+String(i).padStart(6,"0")+
      " подтверждает чтение исходного текста.\n").join("")+
    "\nЗАКЛЮЧЕНИЕ\nЗаключительный правовой вывод.";
const hash=s=>createHash("sha256").update(s,"utf8").digest("hex");
function fixture(){
    const digest=hash(source);
    const first=materializeDocumentOverflowBatch(source,digest,1);
    assert.ok(first);
    const overflowBatches=[];
    for(let i=1;i<=first.batchCount;i++)
        overflowBatches.push(materializeDocumentOverflowBatch(source,digest,i));
    const pages=[];let offset=0;
    do{
        const page=pageDocumentText(source,offset);
        pages.push({...page,name:"paper.docx"});
        offset=page.nextOffset;
    }while(offset!==null);
    return {task:"Сопоставь положения работы",attachments:[
      {name:"paper.docx",pages,overflowBatches}]};
}
async function model(_prompt,_task,attachments){
    const first=createLosslessWindowEvidence(attachments[0]).fragments[0];
    const quote=first.text.split("\n").find(s=>s.trim().length>16)?.trim().slice(0,60);
    assert.ok(quote?.length>16);
    return JSON.stringify({sections:[{heading:"Обоснования",findings:[{
        kind:"author_claim",text:"Автор описывает проверяемые положения работы",
        evidence:[{ref:first.ref,quote}]
    }]}]});
}
async function withKey(action){
    const before=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    const key="b".repeat(64);
    process.env.JESSICA_STAGE_ATTESTATION_KEY=key;
    try{return await action(key);}
    finally{
        if(before===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
        else process.env.JESSICA_STAGE_ATTESTATION_KEY=before;
    }
}
test("verified signed stages continue exactly in sequence and merge on completion",
    async()=>withKey(async key=>{
      const input=fixture();let calls=0;
      const generate=async(...args)=>{calls++;return model(...args);};
      const item=input.attachments[0];
      const first=await analyzeDocumentAttachments(
          {...input,attachments:[{...item,stageNumber:1}]},generate);
      assert.equal(first.success,true,first.stage);
      assert.equal(first.readingCoverage.stageCount,3);
      assert.equal(verifyDocumentStageReceipt(first,key),true);
      assert.equal(calls,16);
      const second=await analyzeDocumentAttachments(
          {...input,attachments:[{...item,stageNumber:2,
              stageReports:[first]}]},generate);
      assert.equal(second.success,true,second.stage);
      assert.equal(second.continuation.precedingStagesVerified,1);
      assert.equal(second.continuation.precedingWindowsVerified,16);
      assert.equal(second.continuation.nextStageNumber,3);
      assert.equal(second.continuation.fullSourceAnalysisConfirmed,false);
      assert.equal(verifyDocumentStageReceipt(second,key),true);
      assert.equal(calls,32);
      const third=await analyzeDocumentAttachments(
          {...input,attachments:[{...item,stageNumber:3,
              stageReports:[first,second]}]},generate);
      assert.equal(third.success,true,third.stage);
      assert.equal(third.continuation.awaitFinalMerge,true);
      assert.equal(third.continuation.nextStageNumber,null);
      assert.equal(verifyDocumentStageReceipt(third,key),true);
      const beforeMerge=calls;
      const merged=await analyzeDocumentAttachments({...input,
          attachments:[{...item,stageReports:[first,second,third]}]},generate);
      assert.equal(calls,beforeMerge);
      assert.equal(merged.success,true,JSON.stringify({stage:merged.stage,
          text:merged.text?.slice(0,130)}));
      assert.equal(merged.stage,"document-stages-authenticated");
      assert.equal(merged.readingCoverage.reportOriginAuthenticated,true);
      assert.equal(merged.readingCoverage.completeSourceAnalysisConfirmed,false);
      assert.equal(merged.readingCoverage.claimTruthVerified,false);
}));
test("unsigned, forged, skipped or mismatched prefixes never call the model",
    async()=>withKey(async()=>{
      const input=fixture(),item=input.attachments[0];
      const first=await analyzeDocumentAttachments(
          {...input,attachments:[{...item,stageNumber:1}]},model);
      assert.equal(first.success,true);
      const tampered=structuredClone(first);
      tampered.stageReport.taskSha256="0".repeat(64);
      const unsigned=structuredClone(first);
      delete unsigned.stageReport.authTag;
      const variants=[
        {stageNumber:2,stageReports:[tampered]},
        {stageNumber:2,stageReports:[unsigned]},
        {stageNumber:3,stageReports:[first]},
        {stageNumber:1,stageReports:[first]},
        {stageNumber:2,stageReports:[]}
      ];
      let calls=0;
      for(const variant of variants){
        const actual=await analyzeDocumentAttachments({...input,
          attachments:[{...item,...variant}]},async()=>{
            calls++;return "{}";
          });
        assert.equal(actual.success,false);
        assert.equal(actual.stage,"document-stage-resume-invalid");
      }
      assert.equal(calls,0);
}));
test("provider quota during continuation never asserts completion",
    async()=>withKey(async()=>{
      const input=fixture(),item=input.attachments[0];
      const first=await analyzeDocumentAttachments({...input,
        attachments:[{...item,stageNumber:1}]},model);
      assert.equal(first.success,true);
      let calls=0;
      const next=await analyzeDocumentAttachments({...input,
        attachments:[{...item,stageNumber:2,stageReports:[first]}]},
        async(...args)=>{
          calls++;
          if(calls===3){const error=new Error("quota");error.status=429;throw error;}
          return model(...args);
        });
      assert.equal(next.success,false);
      assert.equal(next.stage,"ai-provider-cooldown");
      assert.equal(next.readingCoverage.processedWindows,2);
      assert.equal(next.readingCoverage.stageAnalysisComplete,false);
      assert.equal(next.readingCoverage.fullSourceAnalysisConfirmed,false);
      assert.equal(calls,3);
}));
