import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence}
 from "../core/files/documentWindowEvidence.js";
import {planDocumentSectionWindows}
 from "../core/files/documentSectionReadPlan.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {registerAIProviderLimit,getAIProviderAvailability}
 from "../ai/providerAvailability.js";
import {getRetryAfterDetails}
 from "../ai/retry/retryAfter.js";

const source="ВВЕДЕНИЕ\n"+
    "Проверяемое научное утверждение. ".repeat(1150)+
    "\n1.1 Научные основания\n"+
    "Правовая позиция исследователя. ".repeat(1250)+
    "\nЗАКЛЮЧЕНИЕ\nВывод сформулирован.";
const sha=createHash("sha256").update(source,"utf8").digest("hex");
const plan=planDocumentSectionWindows(source);
if(!plan?.complete||plan.windows.length<4)
    throw Error("Synthetic document must have four or more verified windows");
const attachment=(()=>{
    const pages=[];
    for(let offset=0;offset!==null;){
        const page=pageDocumentText(source,offset);
        pages.push({...page,name:"thesis.docx"});
        offset=page.nextOffset;
    }
    return {name:"thesis.docx",pages};
})();
const request={task:"Проверь научные положения",attachments:[attachment]};
const response=async(_prompt,_task,attachments)=>{
    const fragment=createLosslessWindowEvidence(attachments[0])?.fragments[0];
    assert.ok(fragment?.ref);
    return JSON.stringify({sections:[{heading:"Анализ",findings:[{
        kind:"author_claim",text:"Правовая позиция подтверждается цитатой",
        evidence:[{ref:fragment.ref,quote:fragment.text.slice(0,60)}]
    }]}]});
};
const status429=(retry="120")=>{
    const error=new Error("requests per minute exceeded");
    error.status=429;
    error.headers={"retry-after":retry};
    return error;
};
function fakeClock(t,n){
    let ms=Date.now()+n*3_600_000;
    t.mock.method(Date,"now",()=>ms);
    t.mock.method(console,"warn",()=>{});
    return delta=>{ms+=delta;};
}

test("mid-document 429 preserves exact verified-prefix SHA and blocks new calls",async t=>{
    const advance=fakeClock(t,1);
    let calls=0;
    const limited=status429("120");
    assert.deepEqual(getRetryAfterDetails(limited),
        {milliseconds:120000,source:"retry-after"});
    const interrupted=await analyzeDocumentAttachments(request,async(...args)=>{
        calls++;
        if(calls===3)throw limited;
        return response(...args);
    });
    assert.equal(interrupted.success,false);
    assert.equal(interrupted.stage,"ai-provider-cooldown");
    assert.equal(calls,3);
    const coverage=interrupted.readingCoverage;
    assert.equal(coverage.sourceSha256,sha);
    assert.equal(coverage.plannedWindows,plan.windows.length);
    assert.equal(coverage.processedWindows,2);
    assert.equal(coverage.verifiedWindows.length,2);
    assert.equal(coverage.interruptedWindow,3);
    assert.equal(coverage.complete,false);
    assert.equal(coverage.allWindowsPresented,false);
    assert.equal(coverage.resumeAvailable,false);
    assert.equal(coverage.completeSourceAnalysisConfirmed,false);
    assert.equal(interrupted.citationSpans,undefined);
    for(let i=0;i<2;i++){
        const item=coverage.verifiedWindows[i],expected=plan.windows[i];
        assert.equal(item.startOffset,expected.startOffset);
        assert.equal(item.endOffset,expected.endOffset);
        assert.equal(item.windowSha256,
            createHash("sha256").update(expected.text,"utf8").digest("hex"));
    }
    const breaker=getAIProviderAvailability();
    assert.equal(breaker.blocked,true);
    assert.equal(breaker.reason,"long-rate-limit");
    assert.equal(breaker.source,"retry-after");
    assert.equal(breaker.remainingMs,120000);
    const blocked=await analyzeDocumentAttachments(request,async()=>{
        calls++;
        throw Error("Network call during cooldown");
    });
    assert.equal(blocked.stage,"ai-provider-cooldown");
    assert.equal(blocked.success,false);
    assert.equal(calls,3);
    advance(119999);
    assert.equal(getAIProviderAvailability().blocked,true);
    const stillBlocked=await analyzeDocumentAttachments(request,async()=>{
        calls++;throw Error("Premature retry");
    });
    assert.equal(stillBlocked.stage,"ai-provider-cooldown");
    assert.equal(calls,3);
    advance(1);
    assert.equal(getAIProviderAvailability().available,true);
    let fresh=0;
    const repeated=await analyzeDocumentAttachments(request,async(...args)=>{
        fresh++;return response(...args);
    });
    assert.equal(fresh,plan.windows.length);
    assert.equal(repeated.success,true);
    assert.equal(repeated.readingCoverage.sourceSha256,sha);
    assert.equal(repeated.readingCoverage.processedWindows,
        plan.windows.length);
    assert.equal(repeated.readingCoverage.allWindowsPresented,true);
    assert.equal(repeated.readingCoverage.verifiedWindows.length,
        plan.windows.length);
    assert.equal(repeated.citationSpans.length,plan.windows.length);
    for(let i=0;i<repeated.citationSpans.length;i++){
        const span=repeated.citationSpans[i];
        assert.equal(span.ref,"О"+(i+1)+":Ф1:Ф1");
        assert.equal(span.windowNumber,i+1);
        assert.equal(span.startOffset,plan.windows[i].startOffset);
        assert.ok(span.endOffset>span.startOffset);
        assert.equal(span.fragmentSha256,
            createHash("sha256")
                .update(source.slice(span.startOffset,span.endOffset),"utf8")
                .digest("hex"));
    }
    // Fresh request recomputes the document; there is no stored checkpoint.
    assert.equal(repeated.readingCoverage.completeSourceAnalysisConfirmed,false);
    process.stdout.write("DOCUMENT_429_END_TO_END "+
        JSON.stringify({initialModelCalls:calls,blockedRetries:0,
            verifiedPrefix:2,sourceWindows:plan.windows.length,
            restartedFromWindow:1,verifiedCitationSpans:repeated.citationSpans.length,
            simulatedRetryAfterMs:120000,checkpointResume:false})+"\n");
});

test("429 from sibling request stops in-flight document at next window",async t=>{
    fakeClock(t,2);
    let calls=0;
    const partial=await analyzeDocumentAttachments(request,async(...args)=>{
        calls++;
        if(calls===1){
            const opened=registerAIProviderLimit(status429("90"));
            assert.equal(opened.blocked,true);
            assert.equal(opened.remainingMs,90000);
        }
        return response(...args);
    });
    assert.equal(partial.success,false);
    assert.equal(partial.stage,"ai-provider-cooldown");
    assert.equal(calls,1,
        "new window must not call model after global breaker opens");
    assert.equal(partial.readingCoverage.processedWindows,1);
    assert.equal(partial.readingCoverage.verifiedWindows.length,1);
    assert.equal(partial.readingCoverage.interruptedWindow,2);
    assert.equal(partial.readingCoverage.resumeAvailable,false);
    assert.equal(partial.citationSpans,undefined);
});

test("invalid source cannot be mistaken for a successful resumed citation",async t=>{
    fakeClock(t,3);
    let calls=0;
    const damaged=structuredClone(attachment);
    damaged.pages[1].text="ЗАМЕНА"+damaged.pages[1].text.slice(6);
    const result=await analyzeDocumentAttachments({
        task:request.task,attachments:[damaged]
    },async()=>{calls++;return response();});
    assert.equal(result.success,false);
    assert.equal(result.stage,"document-pages");
    assert.equal(calls,0);
    assert.equal(result.citationSpans,undefined);
});
