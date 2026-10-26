/** In-process metadata-only LRU. Source bytes and evidence are never cached. */
const MAX_ENTRIES=2;
const MAX_WINDOWS=256;
const store=new Map();
let hitCount=0,missCount=0;

export function findVerifiedDocumentPlan(sha256,sourceChars){
    if(typeof sha256!=="string"||
       !/^[a-f0-9]{64}$/u.test(sha256)||
       !Number.isSafeInteger(sourceChars)||sourceChars<=0)
        return null;
    const plan=store.get(sha256);
    if(!plan||plan.sourceChars!==sourceChars){
        missCount++;
        return null;
    }
    store.delete(sha256);
    store.set(sha256,plan);
    hitCount++;
    // Never expose the cached plan for caller mutation.
    return structuredClone(plan);
}

export function rememberVerifiedDocumentPlan(plan){
    if(!plan||!/^[a-f0-9]{64}$/u.test(plan.sourceSha256)||
       !Number.isSafeInteger(plan.sourceChars)||plan.sourceChars<=0||
       !Number.isSafeInteger(plan.totalWindows)||
       plan.totalWindows<1||plan.totalWindows>MAX_WINDOWS||
       !Number.isSafeInteger(plan.batchCount)||
       plan.batchCount<1||plan.batchCount>16||
       plan.completeSourcePartition!==true||
       !Array.isArray(plan.batches)||
       plan.batches.length!==plan.batchCount)return false;
    const snapshot=structuredClone(plan);
    store.delete(plan.sourceSha256);
    store.set(plan.sourceSha256,snapshot);
    while(store.size>MAX_ENTRIES)store.delete(store.keys().next().value);
    return true;
}

/** Informational process-local counters, not an analysis of source content. */
export function documentPlanCacheDiagnostics(){
    return {entries:store.size,hits:hitCount,misses:missCount,
        capacity:MAX_ENTRIES,cachesSourceText:false};
}
