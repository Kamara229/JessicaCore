import test from "node:test";
import assert from "node:assert/strict";
import {performance} from "node:perf_hooks";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";
import {documentPlanCacheDiagnostics}
 from "../core/files/documentOverflowPlanCache.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {makeDocumentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";

test("SHA-256 cache reuses 256-window partition without losing source integrity",()=>{
    const source=makeDocumentStressSource();
    const hash=hashDocumentStressSource(source);
    const before=documentPlanCacheDiagnostics();
    const coldStart=performance.now();
    const initial=planDocumentOverflowBatches(source,hash);
    const coldElapsedMs=Math.round(performance.now()-coldStart);
    assert.equal(initial?.totalWindows,256);
    assert.equal(initial.batchCount,16);
    let start=performance.now();
    for(let i=0;i<16;i++){
        const plan=planDocumentOverflowBatches(source,hash);
        assert.equal(plan?.totalWindows,256);
        assert.deepEqual(plan.batches[i],initial.batches[i]);
    }
    const repeated16ElapsedMs=Math.round(performance.now()-start);
    const after=documentPlanCacheDiagnostics();
    assert.equal(after.hits-before.hits,16);
    assert.equal(after.misses-before.misses,1);
    assert.equal(after.entries,1);
    assert.equal(after.cachesSourceText,false);
    assert.equal(after.capacity,2);
    process.stdout.write("DOCUMENT_PLAN_CACHE_COMPARISON "+
        JSON.stringify({coldElapsedMs,repeated16ElapsedMs,
            verifiedCacheHits:after.hits-before.hits,
            measurementScope:"synthetic partition planning, runner-local"})+
        "\n");
});

test("returned metadata is isolated from cache and revalidated on each read",()=>{
    const source=makeDocumentStressSource();
    const hash=hashDocumentStressSource(source);
    const expected=planDocumentOverflowBatches(source,hash);
    expected.batches[0].windows[0].startOffset=12345;
    expected.batches[0].windows[0].title="FORGED";
    expected.batches.length=1;
    const next=planDocumentOverflowBatches(source,hash);
    assert.equal(next.totalWindows,256);
    assert.equal(next.batches.length,16);
    assert.equal(next.batches[0].windows[0].startOffset,0);
    assert.notEqual(next.batches[0].windows[0].title,"FORGED");
    const one=materializeDocumentOverflowBatch(source,hash,1);
    assert.equal(one?.windowCount,16);
    assert.ok(one.windows.every(w=>
        w.text===source.slice(w.startOffset,w.endOffset)));
    assert.equal(planDocumentOverflowBatches(source+"x",hash),null);
    assert.equal(materializeDocumentOverflowBatch(source+"x",hash,1),null);
    assert.equal(planDocumentOverflowBatches(source,"0".repeat(64)),null);
    const bad=[{...one,windows:[...one.windows]}];
    bad[0].windows[0]={...one.windows[0],
        text:one.windows[0].text+"forged"};
    assert.equal(verifyMaterializedOverflowBatches(source,hash,bad),null);
});

test("two-entry eviction keeps metadata cache bounded without skipping validation",()=>{
    const first=makeDocumentStressSource();
    const alt=first.replace("Раздел 1\n","Раздел I\n");
    const third=first.replace("Раздел 2\n","Раздел II\n");
    const h1=hashDocumentStressSource(first);
    const h2=hashDocumentStressSource(alt);
    const h3=hashDocumentStressSource(third);
    assert.equal(planDocumentOverflowBatches(first,h1)?.totalWindows,256);
    assert.equal(planDocumentOverflowBatches(alt,h2)?.totalWindows,256);
    assert.equal(planDocumentOverflowBatches(third,h3)?.totalWindows,256);
    const before=documentPlanCacheDiagnostics();
    assert.equal(before.entries,2);
    const plan=planDocumentOverflowBatches(first,h1);
    const after=documentPlanCacheDiagnostics();
    assert.equal(plan?.totalWindows,256);
    assert.equal(after.entries,2);
    assert.ok(after.misses>before.misses);
    assert.equal(planDocumentOverflowBatches(alt,h1),null);
});
