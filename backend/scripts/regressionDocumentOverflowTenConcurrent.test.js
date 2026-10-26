import test from "node:test";
import assert from "node:assert/strict";
import {performance} from "node:perf_hooks";
import {runSyntheticDocumentRequest}
 from "./fixtures/documentStageClaimStressLifecycle.js";
import {makeConcurrentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";

test("10 concurrent 256-window requests isolate source hashes and 429/cancel",async()=>{
    const before=performance.now();
    let maxRss=process.memoryUsage().rss;
    const sample=setInterval(()=>{
        maxRss=Math.max(maxRss,process.memoryUsage().rss);
    },20);
    let results;
    try{
        // Launch all 10 before awaiting: 8 successful, one synthetic HTTP 429
        // using the actual provider circuit-breaker, one cancellation.
        results=await Promise.all(Array.from({length:10},(_,i)=>{
            const id=i+1;
            return runSyntheticDocumentRequest(id,{
                inject429:id===9,cancelOnStart:id===10
            });
        }));
    }finally{
        clearInterval(sample);
        maxRss=Math.max(maxRss,process.memoryUsage().rss);
    }
    const firstWallMs=Math.round(performance.now()-before);
    assert.equal(results.length,10);
    const completed=results.filter(x=>x.phase==="complete");
    assert.equal(completed.length,8);
    assert.equal(results[8].phase,"rate-limited");
    assert.equal(results[8].status,429);
    assert.equal(results[8].noVerifiedResult,true);
    assert.equal(results[8].modelAnalysisPerformed,false);
    assert.deepEqual(results[9],{
        phase:"cancelled",id:10,noVerifiedResult:true
    });
    const retries=await Promise.all([
        runSyntheticDocumentRequest(9),
        runSyntheticDocumentRequest(10)
    ]);
    assert.ok(retries.every(x=>x.phase==="complete"));
    const successful=[...completed,...retries];
    assert.equal(successful.length,10);
    assert.equal(new Set(successful.map(x=>x.sha)).size,10);
    assert.equal(new Set(successful.map(x=>
        x.originalFirstWindowSha)).size,10);
    assert.equal(new Set(successful.map(x=>x.id)).size,10);
    for(const x of successful){
        const source=makeConcurrentStressSource(x.id);
        assert.equal(x.sha,hashDocumentStressSource(source));
        assert.equal(x.sourceChars,source.length);
        assert.equal(x.windowCount,256);
        assert.equal(x.citationCount,256);
        assert.equal(x.sectionCount,128);
        assert.equal(x.crossPagePairs,1);
        assert.equal(x.semanticConflictVerified,false);
        assert.ok(x.heapMiB<160);
    }
    const wallMs=Math.round(performance.now()-before);
    const rssMiB=Math.round(maxRss/1048576);
    process.stdout.write("DOCUMENT_10_CONCURRENT_RETRY_256_WINDOWS "+
        JSON.stringify({attemptsInitially:10,simulatedHttp429:1,
            interrupted:1,retries:2,completed:10,
            isolatedHashes:10,citationsAudited:2560,
            firstWaveMs:firstWallMs,totalWallMs:wallMs,peakRssMiB:rssMiB,
            scope:"synthetic worker threads, not live provider or Render"})+
        "\n");
    assert.ok(wallMs<180000,"Synthetic load exceeded 180 seconds");
    assert.ok(maxRss<1300*1024*1024,
        "Synthetic parallel RSS > 1300 MiB");
});

test("retry after simulated 429 does not reuse another request's signature",async()=>{
    const limited=await runSyntheticDocumentRequest(10,{inject429:true});
    assert.equal(limited.phase,"rate-limited");
    assert.equal(limited.blocked,true);
    assert.equal(limited.noVerifiedResult,true);
    const clean=await runSyntheticDocumentRequest(10);
    assert.equal(clean.phase,"complete");
    assert.equal(clean.id,10);
    assert.equal(clean.sha,hashDocumentStressSource(
        makeConcurrentStressSource(10)));
    const another=await runSyntheticDocumentRequest(1);
    assert.notEqual(clean.sha,another.sha);
    assert.notEqual(clean.originalFirstWindowSha,
        another.originalFirstWindowSha);
});
