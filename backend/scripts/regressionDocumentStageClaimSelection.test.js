import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {selectAuthenticatedStageClaims}
 from "../core/files/documentStageClaimSelection.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {makeStageClaimFixture,task,secret,source}
 from "./fixtures/documentStageClaimFixture.js";

const select=(params,generate)=>selectAuthenticatedStageClaims({
    ...params,task,secret,source,generate
});
const model=async()=>'{"selectedPositions":[1,3]}';

test("one AI selection returns only original claims and source quotes",async()=>{
 const {verified,reports}=makeStageClaimFixture();
 let calls=0;
 const result=await select({verified,reports,claimOffset:0},async(...args)=>{
    calls++;
    assert.ok(args[0].includes("selectedPositions"));
    assert.ok(args[2][0].text.includes("windowNumber"));
    return model(...args);
 });
 assert.equal(calls,1);
 assert.equal(result.success,true);
 assert.equal(result.stage,"document-stage-claim-selection");
 assert.equal(result.readingCoverage.selectedClaims,2);
 assert.equal(result.readingCoverage.pageClaims,12);
 assert.equal(result.readingCoverage.omittedPageClaims,10);
 assert.equal(result.readingCoverage.newFactGenerationPerformed,false);
 assert.equal(result.readingCoverage.fullDocumentSynthesisPerformed,false);
 assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
 assert.ok(result.text.includes("Окно 1")||result.text.includes("[О1:"));
 assert.ok(result.text.includes("Окно 3")||result.text.includes("[О3:"));
 assert.ok(!result.text.includes("Придуманный результат экспертизы"));
});

test("invalid index lists fail closed without citing invented findings",async()=>{
 const {verified,reports}=makeStageClaimFixture();
 const outputs=[
  '{"selectedPositions":[0]}',
  '{"selectedPositions":[13]}',
  '{"selectedPositions":[2,1]}',
  '{"selectedPositions":[1,1]}',
  '{"selectedPositions":[]}',
  '{"selectedPositions":[1,2,3,4,5,6,7]}',
  '{"selectedPositions":["1"]}',
  '{"selectedPositions":[1],"invention":"fake"}',
  '{"sections":[{"text":"fake"}]}',
  "not json"
 ];
 for(const raw of outputs){
    const result=await select({verified,reports},async()=>raw);
    assert.equal(result.success,false,raw);
    assert.equal(result.stage,"document-stage-claim-selection-invalid",raw);
    assert.equal(result.text?.includes("fake"),false,raw);
 }
});

test("unsigned claims or invalid offsets cannot call the model",async()=>{
 const {verified,reports}=makeStageClaimFixture();
 const unsigned=structuredClone(reports);
 delete unsigned[0].stageReport.authTag;
 let calls=0;
 for(const p of [
  {verified,reports:unsigned,claimOffset:0},
  {verified,reports:reports.slice(0,-1),claimOffset:0},
  {verified,reports,claimOffset:-1},
  {verified,reports,claimOffset:1.5},
  {verified,reports,claimOffset:99999}
 ]){
    const out=await select(p,async()=>{calls++;return model();});
    assert.equal(out,null);
 }
 assert.equal(calls,0);
});

test("overflow facade accepts a signed claim page but never completes the whole document",async()=>{
 const {reports}=makeStageClaimFixture();
 const sha=reports[0].stageReport.sourceSha256;
 const first=materializeDocumentOverflowBatch(source,sha,1);
 const batches=[];
 for(let i=1;i<=first.batchCount;i++)
    batches.push(materializeDocumentOverflowBatch(source,sha,i));
 const pages=[];let cursor=0;
 do{
    const p=pageDocumentText(source,cursor);
    pages.push({...p,name:"paper.docx"});
    cursor=p.nextOffset;
 }while(cursor!==null);
 const old=process.env.JESSICA_STAGE_ATTESTATION_KEY;
 process.env.JESSICA_STAGE_ATTESTATION_KEY=secret;
 try{
    let calls=0;
    const result=await analyzeDocumentAttachments({task,attachments:[{
       name:"paper.docx",pages,overflowBatches:batches,
       stageReports:reports,claimOffset:0
    }]},async()=>{calls++;return '{"selectedPositions":[1]}';});
    assert.equal(calls,1);
    assert.equal(result.success,true);
    assert.equal(result.mode,"document-claim-selection");
    assert.equal(result.readingCoverage.newFactGenerationPerformed,false);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
    const mixed=await analyzeDocumentAttachments({task,attachments:[{
       name:"paper.docx",pages,overflowBatches:batches,
       stageReports:reports,claimOffset:0,stageNumber:2
    }]},async()=>{calls++;return "{}";});
    assert.equal(mixed.success,false);
    assert.equal(mixed.stage,"document-stage-claim-input");
    assert.equal(calls,1);
 }finally{
    if(old===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
    else process.env.JESSICA_STAGE_ATTESTATION_KEY=old;
 }
});
