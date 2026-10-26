import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence}
 from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {planDocumentSectionWindows}
 from "../core/files/documentSectionReadPlan.js";
import {openDocumentPagedResume}
 from "../core/files/documentPagedResumeCheckpoint.js";
import {getAIProviderAvailability}
 from "../ai/providerAvailability.js";

const source="ВВЕДЕНИЕ\n"+
    "Проверяемое научное утверждение. ".repeat(1150)+
    "\n1.1 Научные основания\n"+
    "Правовая позиция исследователя. ".repeat(1250)+
    "\nЗАКЛЮЧЕНИЕ\nВывод сформулирован.";
const task="Проверь научные положения";
const name="thesis.docx";
const digest=text=>createHash("sha256").update(text,"utf8").digest("hex");
const plan=planDocumentSectionWindows(source);
if(!plan?.complete||plan.windows.length!==7)throw Error("Expected seven windows");
function attachment(text=source,fileName=name){
    const pages=[];
    for(let offset=0;offset!==null;){
        const page=pageDocumentText(text,offset);
        pages.push({...page,name:fileName});offset=page.nextOffset;
    }
    return {name:fileName,pages};
}
async function model(_prompt,_task,attachments){
    const fragment=createLosslessWindowEvidence(attachments[0]).fragments[0];
    const quote=fragment.text.slice(0,60);
    return JSON.stringify({sections:[{heading:"Анализ",findings:[{
        kind:"author_claim",text:"Цитата подтверждает позицию",
        evidence:[{ref:fragment.ref,quote}]
    }]}]});
}
function status429(){
    const e=new Error("requests per minute exceeded");
    e.status=429;e.headers={"retry-after":"120"};return e;
}
function withKeyAndClock(t,runner){
    const old=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    process.env.JESSICA_STAGE_ATTESTATION_KEY="a".repeat(64);
    let now=Date.now()+3_600_000;
    t.mock.method(Date,"now",()=>now);
    t.mock.method(console,"warn",()=>{});
    const advance=ms=>{now+=ms;};
    return Promise.resolve().then(()=>runner(advance)).finally(()=>{
        if(old===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
        else process.env.JESSICA_STAGE_ATTESTATION_KEY=old;
    });
}
async function failOnThird(){
    let calls=0;
    const result=await analyzeDocumentAttachments({
        task,attachments:[attachment()]
    },async(...args)=>{
        if(++calls===3)throw status429();
        return model(...args);
    });
    assert.equal(calls,3);
    assert.equal(result.stage,"ai-provider-cooldown");
    assert.equal(result.success,false);
    assert.equal(result.readingCoverage.processedWindows,2);
    assert.equal(result.readingCoverage.resumeAvailable,true);
    assert.equal(result.readingCoverage.resumeCheckpoint?.rawResponses.length,2);
    return result.readingCoverage.resumeCheckpoint;
}

test("signed two-window prefix resumes five missing windows with exact citations",async t=>
 withKeyAndClock(t,async advance=>{
    const checkpoint=await failOnThird();
    const expected=openDocumentPagedResume({
        checkpoint,source,task,name,plan
    });
    assert.equal(expected?.verifiedPrefix,2);
    const [one,two]=checkpoint.windows;
    assert.equal(one.windowSha256,digest(plan.windows[0].text));
    assert.equal(two.windowSha256,digest(plan.windows[1].text));
    assert.equal(checkpoint.sourceSha256,digest(source));
    assert.equal(checkpoint.sourceChars,source.length);
    let calls=0;
    const before=await analyzeDocumentAttachments({
        task,attachments:[{...attachment(),resumeCheckpoint:checkpoint}]
    },async()=>{calls++;throw Error("Request during cooldown");});
    assert.equal(before.stage,"ai-provider-cooldown");
    assert.equal(calls,0);
    advance(120000);
    assert.equal(getAIProviderAvailability().available,true);
    const completed=await analyzeDocumentAttachments({
        task,attachments:[{...attachment(),resumeCheckpoint:checkpoint}]
    },async(...args)=>{calls++;return model(...args);});
    assert.equal(completed.success,true,completed.stage);
    assert.equal(calls,plan.windows.length-2);
    assert.equal(completed.readingCoverage.processedWindows,7);
    assert.equal(completed.readingCoverage.verifiedWindows.length,7);
    assert.equal(completed.readingCoverage.sourceSha256,digest(source));
    assert.equal(completed.citationSpans.length,7);
    assert.equal(completed.readingCoverage.completeSourceAnalysisConfirmed,false);
    for(let i=0;i<7;i++){
        const span=completed.citationSpans[i];
        assert.equal(span.ref,"О"+(i+1)+":Ф1:Ф1");
        assert.equal(span.windowNumber,i+1);
        assert.equal(span.startOffset,plan.windows[i].startOffset);
        assert.equal(span.fragmentSha256,digest(
            source.slice(span.startOffset,span.endOffset)));
    }
    process.stdout.write("DOCUMENT_SIGNED_RESUME "+
        JSON.stringify({confirmedPrefix:2,remainingModelCalls:calls,
            windows:7,verifiedCitations:completed.citationSpans.length,
            modelCallsDuringCooldown:0,sourceSha256Checked:true,
            durableServerStorage:false,semanticTruthVerified:false})+"\n");
 }));

test("tampered checkpoint, changed source/task/file, missing key all fail closed",async t=>
 withKeyAndClock(t,async()=>{
    const checkpoint=await failOnThird();
    const tampered=structuredClone(checkpoint);
    tampered.rawResponses[0]=tampered.rawResponses[0].replace(
        "Цитата подтверждает позицию","Подменённая позиция");
    const alteredWindow=structuredClone(checkpoint);
    alteredWindow.windows[0].startOffset++;
    const alteredSig=structuredClone(checkpoint);
    alteredSig.authTag="v1:"+"f".repeat(64);
    const changed=source.replace("Проверяемое","Подменённое");
    const cases=[
        {task,attachments:[{...attachment(),resumeCheckpoint:tampered}]},
        {task,attachments:[{...attachment(),resumeCheckpoint:alteredWindow}]},
        {task,attachments:[{...attachment(),resumeCheckpoint:alteredSig}]},
        {task,attachments:[{...attachment(changed),resumeCheckpoint:checkpoint}]},
        {task:task+" дополнение",attachments:[{
            ...attachment(),resumeCheckpoint:checkpoint}]},
        {task,attachments:[{...attachment(source,"changed.docx"),
            resumeCheckpoint:checkpoint}]}
    ];
    let calls=0;
    for(const item of cases){
        const actual=await analyzeDocumentAttachments(item,async()=>{
            calls++;throw Error("Unsafe model invocation");
        });
        assert.equal(actual.stage,"document-resume-invalid",
            "Invalid authenticated source/claims must fail before model");
        assert.equal(actual.success,false);
    }
    const before=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
    try{
        const result=await analyzeDocumentAttachments({
            task,attachments:[{...attachment(),resumeCheckpoint:checkpoint}]
        },async()=>{calls++;throw Error("No key, no model");});
        assert.equal(result.stage,"document-resume-invalid");
    }finally{process.env.JESSICA_STAGE_ATTESTATION_KEY=before;}
    assert.equal(calls,0);
 }));

test("a second 429 renews authentic prefix and resumes only remaining windows",async t=>
 withKeyAndClock(t,async advance=>{
    const first=await failOnThird();
    advance(120000);
    let calls=0;
    const interrupted=await analyzeDocumentAttachments({
        task,attachments:[{...attachment(),resumeCheckpoint:first}]
    },async(...args)=>{
        calls++;
        if(calls===2)throw status429();
        return model(...args);
    });
    assert.equal(interrupted.stage,"ai-provider-cooldown");
    assert.equal(calls,2);
    assert.equal(interrupted.readingCoverage.processedWindows,3);
    assert.equal(interrupted.readingCoverage.resumeAvailable,true);
    const second=interrupted.readingCoverage.resumeCheckpoint;
    assert.equal(openDocumentPagedResume({
        checkpoint:second,source,task,name,plan
    }).verifiedPrefix,3);
    advance(120000);
    let lastCalls=0;
    const final=await analyzeDocumentAttachments({
        task,attachments:[{...attachment(),resumeCheckpoint:second}]
    },async(...args)=>{
        lastCalls++;return model(...args);
    });
    assert.equal(final.success,true);
    assert.equal(lastCalls,4);
    assert.equal(final.citationSpans.length,7);
 }));
