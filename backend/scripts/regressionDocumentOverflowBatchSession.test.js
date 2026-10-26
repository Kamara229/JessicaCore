import test from "node:test";
import assert from "node:assert/strict";
import {performance} from "node:perf_hooks";
import {materializeDocumentOverflowBatch,
    prepareDocumentOverflowMaterialization}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {materializeVerifiedDocumentOverflowBatches}
 from "../core/files/documentOverflowBatchBulkMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";
import {makeDocumentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";

const source=makeDocumentStressSource();
const hash=hashDocumentStressSource(source);

test("256-window one-request session retains exact independent batch outputs",()=>{
    const plan=planDocumentOverflowBatches(source,hash);
    assert.equal(plan?.totalWindows,256);
    const baselineStart=performance.now();
    const old=Array.from({length:16},(_,i)=>
        materializeDocumentOverflowBatch(source,hash,i+1));
    const oldMs=Math.round(performance.now()-baselineStart);
    const start=performance.now();
    const session=prepareDocumentOverflowMaterialization(source,hash);
    assert.equal(session?.batchCount,16);
    assert.equal(Object.isFrozen(session),true);
    assert.equal(Object.hasOwn(session,"source"),false);
    const newer=Array.from({length:16},(_,i)=>session.materialize(i+1));
    const sessionMs=Math.round(performance.now()-start);
    assert.deepEqual(newer,old);
    assert.equal(verifyMaterializedOverflowBatches(source,hash,newer)?.complete,
        true);
    assert.equal(session.materialize(0),null);
    assert.equal(session.materialize(17),null);
    assert.equal(session.materialize(Number.NaN),null);
    assert.ok(sessionMs<180000);
    process.stdout.write("DOCUMENT_BATCH_SESSION_SPEED "+
        JSON.stringify({windows:256,independentBatchMs:oldMs,
            sharedSessionBatchMs:sessionMs,scope:"synthetic runner-local"})+
        "\n");
});

test("bulk facade independently checks every source window and citation anchor",()=>{
    const result=materializeVerifiedDocumentOverflowBatches(source,hash);
    assert.equal(result?.independentBatchVerificationPerformed,true);
    assert.equal(result.verified.totalWindows,256);
    assert.equal(result.verified.allSourceWindowsVerified,true);
    assert.equal(result.verified.collectedChars,source.length);
    assert.equal(result.batches.length,16);
    assert.equal(result.modelAnalysisPerformed,false);
    assert.equal(result.semanticCompletenessConfirmed,false);
    for(const [index,window] of result.verified.windows.entries()){
        assert.equal(window.number,index+1);
        assert.equal(source.slice(window.startOffset,window.endOffset),
            window.text);
        assert.equal(hashDocumentStressSource(window.text),window.textSha256);
    }
    const damaged=structuredClone(result.batches);
    damaged[15].windows[15].text+="ПОДМЕНА";
    assert.equal(verifyMaterializedOverflowBatches(source,hash,damaged),null);
    damaged[15].windows[15].text=result.batches[15].windows[15].text;
    damaged[15].windows[15].startOffset++;
    assert.equal(verifyMaterializedOverflowBatches(source,hash,damaged),null);
});

test("changed source invalidates digest trust; no session created for stale hash",()=>{
    const changed=source.replace("Раздел 1\n","Раздел I\n");
    const changedHash=hashDocumentStressSource(changed);
    assert.notEqual(changedHash,hash);
    assert.equal(prepareDocumentOverflowMaterialization(changed,hash),null);
    assert.equal(materializeVerifiedDocumentOverflowBatches(changed,hash),null);
    const next=prepareDocumentOverflowMaterialization(changed,changedHash);
    assert.equal(next?.totalWindows,256);
    assert.equal(next.sourceSha256,changedHash);
    assert.equal(next.materialize(1)?.sourceSha256,changedHash);
    assert.equal(materializeDocumentOverflowBatch(changed,hash,1),null);
    assert.equal(planDocumentOverflowBatches(source,changedHash),null);
});
