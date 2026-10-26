import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";

const sha=t=>createHash("sha256").update(t,"utf8").digest("hex");
const large="ВВЕДЕНИЕ\n"+("Теоретическое положение. ".repeat(11500))+
    "\nГЛАВА 1. Обоснование\n"+
    ("Судебная практика и положения авторов. ".repeat(2750))+
    "\nЗАКЛЮЧЕНИЕ\nОкончательный вывод.";

test("retrieves each of 16-window batches for long thesis without source gaps",()=>{
    const hash=sha(large),plan=planDocumentOverflowBatches(large,hash);
    assert.ok(plan);
    assert.ok(plan.totalWindows>16);
    assert.ok(plan.batchCount>=2);
    let cursor=0,batchNumber=1,globalWindow=0;
    const texts=[];
    while(batchNumber!==null){
        const batch=materializeDocumentOverflowBatch(large,hash,batchNumber);
        assert.ok(batch);
        assert.equal(batch.sourceSha256,hash);
        assert.equal(batch.startOffset,cursor);
        assert.ok(batch.windowCount<=16);
        assert.equal(batch.batchMaterialized,true);
        assert.equal(batch.sourcePartitionVerified,true);
        assert.equal(batch.allBatchesRead,false);
        assert.equal(batch.modelAnalysisPerformed,false);
        assert.equal(batch.analyzedWindows,0);
        assert.equal(batch.resumeImplemented,false);
        assert.equal(batch.semanticCompletenessConfirmed,false);
        for(const window of batch.windows){
            assert.equal(window.number,++globalWindow);
            assert.equal(window.startOffset,cursor);
            assert.equal(window.text,large.slice(window.startOffset,window.endOffset));
            assert.equal(sha(window.text),window.textSha256);
            texts.push(window.text);
            cursor=window.endOffset;
        }
        assert.equal(batch.endOffset,cursor);
        batchNumber=batch.nextBatchNumber;
    }
    assert.equal(cursor,large.length);
    assert.equal(globalWindow,plan.totalWindows);
    assert.equal(texts.join(""),large);
});

test("never accepts altered, missing, malformed or out-of-range document identity",()=>{
    const hash=sha(large);
    assert.equal(materializeDocumentOverflowBatch(large,"0".repeat(64),1),null);
    assert.equal(materializeDocumentOverflowBatch(large+"X",hash,1),null);
    assert.equal(materializeDocumentOverflowBatch(large,hash,0),null);
    assert.equal(materializeDocumentOverflowBatch(large,hash,-1),null);
    assert.equal(materializeDocumentOverflowBatch(large,hash,1.5),null);
    assert.equal(materializeDocumentOverflowBatch(large,hash,999),null);
    assert.equal(materializeDocumentOverflowBatch("",sha(""),1),null);
});

test("text and emoji boundaries are transmitted losslessly",()=>{
    const source="ВВЕДЕНИЕ\n"+"А".repeat(15989)+"🧪"+
        "Б".repeat(265000)+" \r\n";
    const plan=planDocumentOverflowBatches(source,sha(source));
    assert.ok(plan);
    const all=[];
    for(let i=1;i<=plan.batchCount;i++){
        const batch=materializeDocumentOverflowBatch(source,sha(source),i);
        assert.ok(batch);
        for(const window of batch.windows){
            assert.ok(!/[\uD800-\uDBFF]$/u.test(window.text));
            assert.ok(!/^[\uDC00-\uDFFF]/u.test(window.text));
            all.push(window.text);
        }
    }
    assert.equal(all.join(""),source);
});

test("large or structurally unindexable input never creates a misleading batch",()=>{
    const bad="ГЛАВА 1. Необрабатываемый индекс\n"+
        "1.1 Подраздел\nТекст.\n".repeat(132);
    assert.equal(materializeDocumentOverflowBatch(bad,sha(bad),1),null);
    const oversized="ВВЕДЕНИЕ\n"+"Д".repeat(4_200_000);
    assert.equal(materializeDocumentOverflowBatch(oversized,sha(oversized),1),null);
});