import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {beginWindowLedger,noteVerifiedWindow,windowLedgerSummary}
 from "../core/files/documentWindowLedger.js";
import {planDocumentSectionWindows} from "../core/files/documentSectionReadPlan.js";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";

const digest=s=>createHash("sha256").update(s,"utf8").digest("hex");
const source="ВВЕДЕНИЕ\n"+"Значимое положение автора. ".repeat(900)+
 "\n1.1 Правовая позиция\n"+"Юридический материал исследования. ".repeat(700)+
 "\nЗАКЛЮЧЕНИЕ\nАвтор подводит итоги.";
const pagesFor=text=>{
    const pages=[];let at=0;
    do{const p=pageDocumentText(text,at);pages.push({...p,name:"scientific.docx"});at=p.nextOffset;}
    while(at!==null);
    return pages;
};
const checked={success:true,kept:1,rejected:0,citedRefs:["Ф1:Ф1"]};

test("verified reading ledger records ordered offset spans and hashes",()=>{
 const plan=planDocumentSectionWindows(source),hash=digest(source);
 const ledger=beginWindowLedger(plan,hash);
 assert.ok(ledger);
 assert.equal(noteVerifiedWindow(ledger,plan.windows[1],checked),false);
 for(const w of plan.windows)assert.equal(noteVerifiedWindow(ledger,w,checked),true);
 const coverage=windowLedgerSummary(ledger);
 assert.equal(coverage.processedWindows,plan.windows.length);
 assert.equal(coverage.allWindowsPresented,true);
 assert.equal(coverage.exactTextCoverageConfirmed,true);
 assert.equal(coverage.completeSourceAnalysisConfirmed,false);
 assert.equal(coverage.verifiedWindows[0].windowSha256,digest(plan.windows[0].text));
 assert.equal(JSON.stringify(coverage).includes("Юридический материал исследования"),false);
});

test("tampered text, missing window, and invalid analysis never mark a window as read",()=>{
 const plan=planDocumentSectionWindows(source),hash=digest(source);
 assert.equal(beginWindowLedger(plan,"e".repeat(64)),null);
 assert.equal(beginWindowLedger({...plan,windows:plan.windows.slice(1)},hash),null);
 const ledger=beginWindowLedger(plan,hash);
 assert.equal(noteVerifiedWindow(ledger,{...plan.windows[0],text:"X"},checked),false);
 assert.equal(noteVerifiedWindow(ledger,plan.windows[0],{...checked,kept:0}),false);
 assert.equal(windowLedgerSummary(ledger).processedWindows,0);
});

test("paged document analysis returns per-window evidence coverage",async()=>{
 const plan=planDocumentSectionWindows(source);
 let invoked=0;
 const result=await analyzeDocumentAttachments({
    task:"Проанализируй",attachments:[{name:"scientific.docx",pages:pagesFor(source)}]
 },async(_prompt,_task,attachments)=>{
    invoked++;
    const evidence=createLosslessWindowEvidence(attachments[0]);
    return JSON.stringify({sections:[{heading:"Анализ",findings:[{
      text:"Автор приводит научное положение",kind:"author_claim",
      evidence:[{ref:"Ф1:Ф1",quote:evidence.fragments[0].text.slice(0,60)}]
    }]}]});
 });
 assert.equal(result.success,true);
 assert.equal(invoked,plan.windows.length);
 assert.equal(result.readingCoverage.verifiedWindows.length,invoked);
 assert.equal(result.readingCoverage.sourceSha256,digest(source));
 assert.equal(result.readingCoverage.allWindowsPresented,true);
 assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});

test("failure in later window leaves ledger incomplete",async()=>{
 let seen=0;
 const result=await analyzeDocumentAttachments({
    task:"Проверь",attachments:[{name:"scientific.docx",pages:pagesFor(source)}]
 },async(_prompt,_task,attachments)=>{
    seen++;
    if(seen===2)return '{"sections":[]}';
    const evidence=createLosslessWindowEvidence(attachments[0]);
    return JSON.stringify({sections:[{heading:"Анализ",findings:[{
      text:"Автор приводит позицию",kind:"author_claim",
      evidence:[{ref:"Ф1:Ф1",quote:evidence.fragments[0].text.slice(0,60)}]
    }]}]});
 });
 assert.equal(result.success,false);
 assert.equal(result.stage,"document-window-evidence");
 assert.equal(result.readingCoverage.processedWindows,1);
 assert.equal(result.readingCoverage.allWindowsPresented,false);
});
