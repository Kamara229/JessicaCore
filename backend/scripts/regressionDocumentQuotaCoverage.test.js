import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {markDocumentQuotaInterruption} from "../core/files/documentQuotaCoverage.js";

const pages=(text,name)=>{
    const result=[];let offset=0;
    do{
        const page=pageDocumentText(text,offset);
        result.push({...page,name});offset=page.nextOffset;
    }while(offset!==null);
    return {name,pages:result};
};
const sample="ВВЕДЕНИЕ\n"+"Проверяемое научное утверждение. ".repeat(1150)+
    "\n1.1 Научные основания\n"+"Правовая позиция исследователя. ".repeat(1250)+
    "\nЗАКЛЮЧЕНИЕ\nВывод сформулирован.";
const response=async(_,task,attachments)=>{
    const fragment=createLosslessWindowEvidence(attachments[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Содержание",findings:[{
        kind:"author_claim",text:"Автор рассматривает научный вопрос",
        evidence:[{ref:fragment.ref,quote:fragment.text.slice(0,60)}]
    }]}]});
};
const quota=()=>{
    const error=new Error("429 quota reached");
    error.status=429;
    return error;
};

test("a 429 after verified windows reports exact read progress without false success",async()=>{
    const plan=planDocumentSectionWindows(sample);
    assert.ok(plan.windows.length>3);
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Проанализируй",
        attachments:[pages(sample,"thesis.docx")]},async(...args)=>{
        calls++;
        if(calls===3)throw quota();
        return response(...args);
    });
    assert.equal(result.success,false);
    assert.equal(result.stage,"ai-provider-cooldown");
    assert.equal(calls,3);
    assert.equal(result.readingCoverage.plannedWindows,plan.windows.length);
    assert.equal(result.readingCoverage.processedWindows,2);
    assert.equal(result.readingCoverage.verifiedWindows.length,2);
    assert.equal(result.readingCoverage.interruptedWindow,3);
    assert.equal(result.readingCoverage.complete,false);
    assert.equal(result.readingCoverage.allWindowsPresented,false);
    assert.equal(result.readingCoverage.resumeAvailable,false);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(result.citationSpans,undefined);
});

test("a 429 on the first window correctly reports zero confirmation",async()=>{
    const result=await analyzeDocumentAttachments({task:"Проанализируй",
        attachments:[pages(sample,"thesis.docx")]},async()=>{throw quota();});
    assert.equal(result.stage,"ai-provider-cooldown");
    assert.equal(result.readingCoverage.processedWindows,0);
    assert.equal(result.readingCoverage.verifiedWindows.length,0);
    assert.equal(result.readingCoverage.interruptedWindow,1);
});

test("multiple files preserve completed file and current-file progress on 429",async()=>{
    const a="ВВЕДЕНИЕ\n"+"Первый научный вывод и основания. ".repeat(800);
    const b="ВВЕДЕНИЕ\n"+"Второй научный вывод и основания. ".repeat(930)+
        "\nЗАКЛЮЧЕНИЕ\nОбщее положение.";
    const aWindows=planDocumentSectionWindows(a).windows.length;
    const bWindows=planDocumentSectionWindows(b).windows.length;
    assert.ok(aWindows>=2 && bWindows>=2);
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Сопоставь",
        attachments:[pages(a,"same.docx"),pages(b,"same.docx")]},async(...args)=>{
        calls++;
        if(calls===aWindows+2)throw quota();
        return response(...args);
    });
    assert.equal(result.success,false);
    assert.equal(result.stage,"ai-provider-cooldown");
    assert.equal(result.readingCoverage.plannedWindows,aWindows+bWindows);
    assert.equal(result.readingCoverage.processedWindows,aWindows+1);
    assert.equal(result.readingCoverage.files.length,1);
    assert.equal(result.readingCoverage.incompleteFile.fileIndex,2);
    assert.equal(result.readingCoverage.incompleteFile.processedWindows,1);
    assert.equal(result.readingCoverage.incompleteFile.interruptedWindow,2);
    assert.equal(result.readingCoverage.allWindowsPresented,false);
    assert.equal(result.readingCoverage.resumeAvailable,false);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});

test("nonquota failures are never labeled as provider rate limiting",async()=>{
    let calls=0;
    const result=await analyzeDocumentAttachments({task:"Проверь",
        attachments:[pages(sample,"thesis.docx")]},async(...args)=>{
        calls++;
        if(calls===2)throw Error("unexpected model error");
        return response(...args);
    });
    assert.equal(result.success,false);
    assert.equal(result.stage,"document-window-error");
    assert.equal(result.readingCoverage.processedWindows,1);
    assert.equal(result.readingCoverage.interruption,undefined);
});

test("metadata is only attached to genuine 429 errors",()=>{
    const err=new Error("invalid answer");
    err.status=400;
    assert.equal(markDocumentQuotaInterruption(err,{processedWindows:1},2),err);
    assert.equal(err.documentReadingCoverage,undefined);
});
