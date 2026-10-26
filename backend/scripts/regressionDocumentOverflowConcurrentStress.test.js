import test from "node:test";
import assert from "node:assert/strict";
import {Worker} from "node:worker_threads";
import {performance} from "node:perf_hooks";
import {makeConcurrentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";
import {materializeVerifiedDocumentOverflowBatches}
 from "../core/files/documentOverflowBatchBulkMaterialization.js";
import {documentPlanCacheDiagnostics}
 from "../core/files/documentOverflowPlanCache.js";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";

const workerUrl=new URL("./fixtures/documentStageClaimConcurrencyWorker.js",
    import.meta.url);

function executeWorker(id,{cancelOnStart=false}={}){
    return new Promise((resolve,reject)=>{
        const worker=new Worker(workerUrl,{workerData:{id},
            resourceLimits:{maxOldGenerationSizeMb:160}});
        let started=false,completed=null;
        worker.on("message",msg=>{
            if(msg?.id!==id){
                worker.terminate();
                reject(Error("Cross-request worker identity mismatch"));
                return;
            }
            if(msg.phase==="started"){
                started=true;
                if(cancelOnStart)worker.terminate().catch(reject);
            }else if(msg.phase==="complete"){
                completed=msg;
            }else{
                worker.terminate();
                reject(Error("Unexpected worker payload"));
            }
        });
        worker.once("error",reject);
        worker.once("exit",code=>{
            if(cancelOnStart){
                if(!started||completed!==null||code===0)
                    reject(Error("Cancelled request leaked or finished output"));
                else resolve({cancelled:true,id,noResult:true});
            }else if(code!==0||!completed||!started)
                reject(Error("Incomplete worker request id="+id+
                    " exit="+code));
            else resolve(completed);
        });
    });
}

test("8 concurrent 256-window requests keep SHA, citations and sections isolated",async()=>{
    const began=performance.now();
    let maximumRss=process.memoryUsage().rss;
    const timer=setInterval(()=>{
        maximumRss=Math.max(maximumRss,process.memoryUsage().rss);
    },25);
    let results;
    try{
        // True OS threads: 4 simultaneous requests per wave, 8 total.
        const [one,two]=await Promise.all([
            Promise.all([1,2,3,4].map(id=>executeWorker(id))),
            Promise.all([5,6,7,8].map(id=>executeWorker(id)))
        ]);
        results=[...one,...two];
    }finally{
        clearInterval(timer);
        maximumRss=Math.max(maximumRss,process.memoryUsage().rss);
    }
    const wallMs=Math.round(performance.now()-began);
    assert.equal(results.length,8);
    assert.equal(new Set(results.map(x=>x.sha)).size,8);
    assert.equal(new Set(results.map(x=>x.originalFirstWindowSha)).size,8);
    for(const output of results){
        const source=makeConcurrentStressSource(output.id);
        assert.equal(output.phase,"complete");
        assert.equal(output.sha,hashDocumentStressSource(source));
        assert.equal(output.sourceChars,source.length);
        assert.equal(output.windowCount,256);
        assert.equal(output.citationCount,256);
        assert.equal(output.sectionCount,128);
        assert.equal(output.crossPagePairs,1);
        assert.equal(output.semanticConflictVerified,false);
        assert.ok(output.heapMiB<160);
    }
    const maxRssMiB=Math.round(maximumRss/1048576);
    process.stdout.write("DOCUMENT_8_CONCURRENT_256_WINDOWS "+
        JSON.stringify({requests:8,maxConcurrentWorkers:8,
            windowsPerRequest:256,wallMs,maxRssMiB,
            isolatedHashes:8,citationsAudited:2048,
            scope:"worker_threads, synthetic fixture, not Render"})+"\n");
    assert.ok(wallMs<180000,"Concurrent synthetic stress > 180 seconds");
    assert.ok(maximumRss<1300*1024*1024,
        "Concurrent worker RSS exceeded 1.3 GiB");
});

test("cancelled 256-window request emits no verified result; retry succeeds",async()=>{
    const interrupted=await executeWorker(9,{cancelOnStart:true});
    assert.deepEqual(interrupted,{cancelled:true,id:9,noResult:true});
    const retry=await executeWorker(9);
    assert.equal(retry?.phase,"complete");
    assert.equal(retry.windowCount,256);
    assert.equal(retry.citationCount,256);
    assert.equal(retry.sha,hashDocumentStressSource(
        makeConcurrentStressSource(9)));
});

test("source-hash mismatch and LRU eviction never cross document identities",()=>{
    const inputs=[1,2,3].map(makeConcurrentStressSource);
    const hashes=inputs.map(hashDocumentStressSource);
    for(let i=0;i<3;i++){
        const plan=planDocumentOverflowBatches(inputs[i],hashes[i]);
        assert.equal(plan?.sourceSha256,hashes[i]);
        assert.equal(plan.totalWindows,256);
    }
    assert.ok(documentPlanCacheDiagnostics().entries<=2);
    for(let i=0;i<3;i++){
        assert.equal(planDocumentOverflowBatches(inputs[i],
            hashes[(i+1)%3]),null);
    }
    assert.equal(materializeVerifiedDocumentOverflowBatches(
        inputs[0],hashes[1]),null);
    const first=planDocumentOverflowBatches(inputs[0],hashes[0]);
    assert.equal(first?.totalWindows,256);
    assert.equal(first.sourceSha256,hashes[0]);
    assert.notEqual(first.sourceSha256,hashes[1]);
});
