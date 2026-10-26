import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import {once} from "node:events";
import {createHash} from "node:crypto";
import {registerJessicaJsonParser}
 from "../http/routes/jsonParser.js";
import {registerSolveRoute} from "../http/routes/solveRoute.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {verifyDocumentStageReceipt}
 from "../core/files/documentStageReceiptAuth.js";
import {getAIProviderAvailability}
 from "../ai/providerAvailability.js";
import {httpAttachment,httpTask,httpSource,httpSourceSha256,
    syntheticDocumentAnswer}
 from "./fixtures/documentStage256HttpFixture.js";

const hash=s=>createHash("sha256").update(s,"utf8").digest("hex");
const URL_TOKEN="synthetic-ci-token-not-a-secret";
const STAGE_KEY="a".repeat(64);

test("real authorized /api/solve transports signed 256-window HTTP 429 resume",async t=>{
    const priorToken=process.env.JESSICA_APP_TOKEN;
    const priorSigningKey=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    const priorDebug=process.env.JESSICA_CHAT_DEBUG_ENABLED;
    process.env.JESSICA_APP_TOKEN=URL_TOKEN;
    process.env.JESSICA_STAGE_ATTESTATION_KEY=STAGE_KEY;
    process.env.JESSICA_CHAT_DEBUG_ENABLED="false";
    let now=Date.now()+3600000;
    t.mock.method(Date,"now",()=>now);
    t.mock.method(console,"warn",()=>{});
    let callCount=0,blockedExtraCalls=0;
    const provider=async(...args)=>{
        callCount++;
        if(callCount===243){
            const error=new Error("requests per minute exceeded");
            error.status=429;
            error.headers={"retry-after":"120"};
            throw error;
        }
        return syntheticDocumentAnswer(...args);
    };
    const app=express();
    // Production JSON parser, production authenticated route, production
    // file analysis facade. ONLY the external Groq provider is simulated.
    registerJessicaJsonParser(app);
    registerSolveRoute(app,{executeTask:async(task,{attachments})=>{
        const result=await analyzeDocumentAttachments({task,attachments},
            provider);
        return {...result,engine:"jessica-core"};
    }});
    app.post("/api/ordinary-json-check",(req,res)=>
        res.json({success:true,bytes:JSON.stringify(req.body).length}));
    const server=app.listen(0,"127.0.0.1");
    try{
        await once(server,"listening");
        const baseUrl="http://127.0.0.1:"+server.address().port;
        const send=async(payload,token=URL_TOKEN)=>{
            const response=await fetch(baseUrl+"/api/solve",{
                method:"POST",
                headers:{"content-type":"application/json",
                    "x-jessica-token":token},
                body:JSON.stringify(payload),
                signal:AbortSignal.timeout(30000)
            });
            const body=await response.json();
            return {code:response.status,
                noStore:response.headers.get("cache-control"),
                body};
        };
        const payload=(stageProps,task=httpTask)=>({
            task,outputLanguage:"ru",
            attachments:[{...httpAttachment,...stageProps}]
        });
        const serialized=JSON.stringify(payload({stageNumber:1}));
        // This is an 8+MiB production-shaped request: a former 1MiB
        // global express.json cap rejected it before document parsing.
        assert.ok(Buffer.byteLength(serialized,"utf8")>1024*1024);
        assert.ok(Buffer.byteLength(serialized,"utf8")<12*1024*1024);
        const unauthorized=await send(payload({stageNumber:1}),"wrong");
        assert.equal(unauthorized.code,401);
        assert.equal(unauthorized.body.success,false);
        assert.equal(callCount,0);
        const reports=[];
        for(let number=1;number<=15;number++){
            const {code,noStore,body}=await send(payload({
                stageNumber:number,
                ...(number>1?{stageReports:reports}:{})
            }));
            assert.equal(code,200);
            assert.equal(noStore,"no-store");
            assert.equal(body.success,true,JSON.stringify({
                number,stage:body.stage,text:body.text?.slice(0,140)}));
            assert.equal(body.stage,"document-stage-verified");
            assert.equal(body.readingCoverage.processedWindows,16);
            assert.equal(body.stageReport.sourceSha256,httpSourceSha256);
            assert.equal(verifyDocumentStageReceipt(body,STAGE_KEY),true);
            reports.push(body);
        }
        assert.equal(callCount,240);
        const failed=await send(payload({
            stageNumber:16,stageReports:reports
        }));
        assert.equal(failed.code,200); // JSON status contract, provider 429
        assert.equal(failed.body.success,false);
        assert.equal(failed.body.stage,"ai-provider-cooldown");
        assert.equal(failed.body.readingCoverage.interruptedWindow,243);
        assert.equal(failed.body.readingCoverage.processedWindows,2);
        assert.equal(failed.body.readingCoverage.resumeAvailable,true);
        assert.equal(failed.body.readingCoverage.fullSourceAnalysisConfirmed,false);
        const checkpoint=failed.body.readingCoverage.resumeCheckpoint;
        assert.equal(checkpoint.sourceSha256,httpSourceSha256);
        assert.deepEqual(checkpoint.windows.map(x=>x.number),[241,242]);
        assert.equal(callCount,243);
        assert.equal(getAIProviderAvailability().remainingMs,120000);
        const blocked=await send(payload({
            stageNumber:16,stageReports:reports,resumeCheckpoint:checkpoint
        }));
        assert.equal(blocked.code,200);
        assert.equal(blocked.body.stage,"ai-provider-cooldown");
        assert.equal(callCount,243);
        blockedExtraCalls=callCount-243;
        now+=120000;
        assert.equal(getAIProviderAvailability().available,true);
        const forged=structuredClone(checkpoint);
        forged.windows[0].windowSha256="0".repeat(64);
        const invalid=await send(payload({
            stageNumber:16,stageReports:reports,resumeCheckpoint:forged
        }));
        assert.equal(invalid.code,200);
        assert.equal(invalid.body.success,false);
        assert.equal(invalid.body.stage,"document-stage-resume-invalid");
        assert.equal(callCount,243);
        const resumed=await send(payload({
            stageNumber:16,stageReports:reports,resumeCheckpoint:checkpoint
        }));
        assert.equal(resumed.code,200);
        assert.equal(resumed.body.success,true,resumed.body.stage);
        assert.equal(callCount,257); // Only 14 missing windows were sent.
        assert.equal(resumed.body.readingCoverage.verifiedWindows[0].windowNumber,241);
        assert.equal(resumed.body.readingCoverage.verifiedWindows.at(-1).windowNumber,256);
        assert.equal(verifyDocumentStageReceipt(resumed.body,STAGE_KEY),true);
        const completed=await send(payload({
            stageReports:[...reports,resumed.body]
        }));
        assert.equal(completed.code,200);
        assert.equal(completed.body.stage,"document-stages-authenticated");
        assert.equal(completed.body.success,true);
        assert.equal(completed.body.readingCoverage.processedWindows,256);
        assert.equal(completed.body.readingCoverage.authenticatedStages,16);
        assert.equal(completed.body.readingCoverage.reportOriginAuthenticated,true);
        assert.equal(completed.body.readingCoverage.reportBodySourceQuotesChecked,true);
        assert.equal(completed.body.readingCoverage.completeSourceAnalysisConfirmed,false);
        assert.equal(callCount,257);
        const changedSource=structuredClone(httpAttachment);
        changedSource.pages[0].text="Изменение"+changedSource.pages[0].text.slice(9);
        const mismatch=await send({
            task:httpTask,attachments:[{...changedSource,
                stageNumber:16,stageReports:reports,
                resumeCheckpoint:checkpoint}]
        });
        assert.equal(mismatch.body.success,false);
        assert.equal(mismatch.body.stage,"document-overflow-batches-invalid");
        assert.equal(callCount,257);
        const wrongTask=await send(payload({
            stageNumber:16,stageReports:reports,resumeCheckpoint:checkpoint
        },httpTask+" другое"));
        assert.equal(wrongTask.body.stage,"document-stage-resume-invalid");
        assert.equal(callCount,257);
        assert.equal(hash(httpSource),httpSourceSha256);
        process.stdout.write("DOCUMENT_256_HTTP_RESUME "+
            JSON.stringify({localHttp:true,productionSolveRoute:true,
                externalProviderSimulated:true,jsonBytes:Buffer.byteLength(serialized),
                sourceWindows:256,signedStages:16,
                interruptedAt:243,verifiedPrefix:242,
                replayedStageWindows:2,missingModelCalls:14,
                suppressedDuringCooldown:blockedExtraCalls,
                modelCalls:callCount,authenticatedMerge:true,
                semanticCorrectnessVerified:false})+"\n");
    }finally{
        await new Promise(resolve=>server.close(resolve));
        if(priorToken===undefined)delete process.env.JESSICA_APP_TOKEN;
        else process.env.JESSICA_APP_TOKEN=priorToken;
        if(priorSigningKey===undefined)
            delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
        else process.env.JESSICA_STAGE_ATTESTATION_KEY=priorSigningKey;
        if(priorDebug===undefined)delete process.env.JESSICA_CHAT_DEBUG_ENABLED;
        else process.env.JESSICA_CHAT_DEBUG_ENABLED=priorDebug;
    }
});
