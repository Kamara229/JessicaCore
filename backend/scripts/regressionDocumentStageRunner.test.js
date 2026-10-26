import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {documentStageTaskSha256} from "../core/files/documentStageTaskIdentity.js";
import {verifyDocumentStageReceipt} from "../core/files/documentStageReceiptAuth.js";
import {hashDocumentStageReportText} from "../core/files/documentStageReportTextHash.js";
import {hasLiteralDocumentStageQuotes} from "../core/files/documentStageLiteralQuotes.js";

const sha=t=>createHash("sha256").update(t,"utf8").digest("hex");
const source="ВВЕДЕНИЕ\n"+"Норма права и судебная практика. ".repeat(19500)+
    "\nЗАКЛЮЧЕНИЕ\nОсновные выводы.";
function fixture(){
 const hash=sha(source),first=materializeDocumentOverflowBatch(source,hash,1);
 assert.ok(first);
 const overflowBatches=[];
 for(let n=1;n<=first.batchCount;n++)
   overflowBatches.push(materializeDocumentOverflowBatch(source,hash,n));
 const pages=[];let offset=0;
 do{const p=pageDocumentText(source,offset);pages.push({...p,name:"thesis.docx"});
    offset=p.nextOffset;}while(offset!==null);
 return {task:"Проверь диссертацию",attachments:[
   {name:"thesis.docx",pages,overflowBatches,stageNumber:1}]};
}
const valid=async(_prompt,_task,files)=>{
 const f=createLosslessWindowEvidence(files[0]).fragments[0];
 return JSON.stringify({sections:[{heading:"Материал",findings:[{
    kind:"author_claim",text:"Автор обосновывает правовую позицию",
    evidence:[{ref:f.ref,quote:f.text.slice(0,55)}]
 }]}]});
};
test("one trusted stage processes 16 windows without asserting full-file analysis",async()=>{
 const req=fixture(),seen=[];
 const result=await analyzeDocumentAttachments(req,async(...args)=>{
    seen.push(args[2][0].text);return valid(...args);
 });
 assert.equal(result.success,true,JSON.stringify({stage:result.stage,text:result.text?.slice(0,80)}));
 assert.equal(result.stage,"document-stage-verified");
 assert.equal(result.mode,"document-stage");
 assert.equal(seen.length,16);
 assert.equal(seen.join(""),source.slice(0,seen.join("").length));
 assert.equal(result.stageReport.stageAnalysisComplete,true);
 assert.equal(result.stageReport.taskSha256,documentStageTaskSha256(req.task));
 assert.equal(result.stageReport.reportTextSha256,hashDocumentStageReportText(result.text));
 assert.equal(result.readingCoverage.taskSha256,documentStageTaskSha256(req.task));
 assert.equal(result.stageReport.fullSourceAnalysisConfirmed,false);
 assert.equal(result.readingCoverage.stageAnalysisComplete,true);
 assert.ok(result.readingCoverage.totalSourceWindows>32);
 assert.equal(result.readingCoverage.processedWindows,16);
 assert.equal(result.readingCoverage.verifiedWindows.length,16);
 assert.ok(result.readingCoverage.verifiedWindows.every(w=>w.anchoredFragments>0));
 assert.ok(result.readingCoverage.verifiedWindows.every(w=>w.claimTruthVerified===false));
});
test("stage model output receives optional server-only attestation",async()=>{
 const key="c".repeat(64),previous=process.env.JESSICA_STAGE_ATTESTATION_KEY;
 try{
    process.env.JESSICA_STAGE_ATTESTATION_KEY=key;
    const result=await analyzeDocumentAttachments(fixture(),valid);
    assert.equal(result.success,true);
    assert.equal(verifyDocumentStageReceipt(result,key),true);
    assert.equal(verifyDocumentStageReceipt(result,"d".repeat(64)),false);
    const forged=structuredClone(result);
    forged.readingCoverage.verifiedWindows[0].groundedClaims+=1;
    assert.equal(verifyDocumentStageReceipt(forged,key),false);
 }finally{
    if(previous===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
    else process.env.JESSICA_STAGE_ATTESTATION_KEY=previous;
 }
});

test("stage citations require literal quotes, not merely normalized matches",async()=>{
 const request=fixture();
 let calls=0;
 const result=await analyzeDocumentAttachments(request,async(_prompt,_task,files)=>{
   calls++;
   const fragment=createLosslessWindowEvidence(files[0]).fragments[0];
   const literal=fragment.text.slice(0,55);
   const normalized=literal.replace(/\n/gu," ");
   assert.notEqual(normalized,literal);
   return JSON.stringify({sections:[{heading:"Основания",findings:[{
      kind:"author_claim",text:"Автор рассматривает правовые основания",
      evidence:[{ref:fragment.ref,quote:normalized}]
   }]}]});
 });
 assert.equal(calls,1);
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-stage-window-evidence");
 assert.equal(result.readingCoverage.processedWindows,0);
});

test("literal but repeated quotes are not mislabeled as unique offsets",()=>{
 const literal={evidenceQuotes:2,exactLocatedQuotes:1,
    ambiguousQuotes:1,nonLiteralQuotes:0};
 assert.equal(hasLiteralDocumentStageQuotes(literal),true);
 assert.equal(hasLiteralDocumentStageQuotes({...literal,nonLiteralQuotes:1}),false);
 assert.equal(hasLiteralDocumentStageQuotes({...literal,exactLocatedQuotes:0}),false);
});

test("unsupported model claims cannot complete stage",async()=>{
 const req=fixture();let calls=0;
 const result=await analyzeDocumentAttachments(req,async(...args)=>{
    calls++;if(calls===3)return '{"sections":[]}';return valid(...args);
 });
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-stage-window-evidence");
 assert.equal(result.readingCoverage.processedWindows,2);
 assert.equal(result.readingCoverage.stageAnalysisComplete,false);
 assert.equal(result.readingCoverage.fullSourceAnalysisConfirmed,false);
 assert.equal(calls,3);
});
test("invalid stage and absent batches never invoke a model",async()=>{
 const req=fixture();let calls=0;
 for(const change of [
   {...req.attachments[0],stageNumber:999},
   {...req.attachments[0],stageNumber:0},
   {...req.attachments[0],overflowBatches:req.attachments[0].overflowBatches.slice(0,1)}
 ]){
   const result=await analyzeDocumentAttachments({...req,attachments:[change]},
      async()=>{calls++;return "{}";});
   assert.equal(result.success,false);
 }
 assert.equal(calls,0);
});
test("429 preserves partial stage progress, never reporting a successful test",async()=>{
 const req=fixture();let calls=0;
 const result=await analyzeDocumentAttachments(req,async(...args)=>{
    calls++;
    if(calls===3){const err=new Error("provider quota");err.status=429;throw err;}
    return valid(...args);
 });
 assert.equal(result.success,false);
 assert.equal(result.stage,"ai-provider-cooldown");
 assert.equal(calls,3);
 assert.equal(result.readingCoverage.processedWindows,2);
 assert.equal(result.readingCoverage.interruptedWindow,3);
 assert.equal(result.readingCoverage.stageAnalysisComplete,false);
 assert.equal(result.readingCoverage.fullSourceAnalysisConfirmed,false);
});
