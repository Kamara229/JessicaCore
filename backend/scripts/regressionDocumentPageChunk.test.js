import test from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {buildDocumentPageChunk} from "../core/files/documentPageChunk.js";
import {buildDocumentPageBatch} from "../core/files/documentPageBatch.js";
import {assembleDocumentPages} from "../core/files/documentPageAssembly.js";

const sha=text=>createHash("sha256").update(text,"utf8").digest("hex");
const large="ВВЕДЕНИЕ\n"+"Правовое положение. ".repeat(17500)+
    "\nГЛАВА 1.\n"+"Другой научный аргумент. ".repeat(3800)+"\n\n";

test("large document is transported in consecutive verified batches without omission",()=>{
    assert.equal(buildDocumentPageBatch(large,"thesis.docx"),null);
    let offset=0,previousSha=null,chunks=0;
    const all=[];
    while(offset!==null){
        const batch=buildDocumentPageChunk(large,"thesis.docx",offset,previousSha);
        assert.ok(batch);
        assert.equal(batch.startOffset,offset);
        assert.equal(batch.pageCount,batch.pages.length);
        assert.ok(batch.pageCount<=16);
        assert.equal(batch.sourceSha256,sha(large));
        assert.equal(batch.modelAnalysisPerformed,false);
        assert.equal(batch.semanticCompletenessConfirmed,false);
        if(chunks===0)assert.equal(batch.sourceFullyCollected,false);
        all.push(...batch.pages);
        offset=batch.nextOffset;
        previousSha=batch.sourceSha256;
        if(batch.hasMore)assert.equal(batch.endOffset,offset);
        else assert.equal(batch.endOffset,large.length);
        chunks++;
        assert.ok(chunks<=16);
    }
    assert.ok(chunks>=2);
    assert.equal(all.length>16,true);
    assert.equal(assembleDocumentPages(all).text,large);
});

test("continuation requires same source hash; damaged pages fail final assembly",()=>{
    const first=buildDocumentPageChunk(large,"work.pdf");
    const second=buildDocumentPageChunk(large,"work.pdf",first.nextOffset,first.sourceSha256);
    assert.ok(second);
    assert.equal(buildDocumentPageChunk(large+"X","work.pdf",first.nextOffset,
        first.sourceSha256),null);
    assert.equal(buildDocumentPageChunk(large,"work.pdf",first.nextOffset,
        "0".repeat(64)),null);
    assert.equal(buildDocumentPageChunk(large,"work.pdf",1.2),null);
    assert.equal(buildDocumentPageChunk(large,"work.pdf",-1),null);
    assert.equal(buildDocumentPageChunk(large,"work.pdf",large.length),null);
    assert.equal(assembleDocumentPages([...first.pages,...second.pages].map((page,i)=>
        i===1?{...page,text:"Z"+page.text.slice(1)}:page)),null);
});

test("surrogate pairs and UTF-8 response byte budget survive chunk boundaries",()=>{
    const unicode="🧪".repeat(145000)+" \r\n";
    let offset=0,previousSha=null,chunks=0;
    const pages=[];
    do{
        const batch=buildDocumentPageChunk(unicode,"unicode.txt",offset,previousSha);
        assert.ok(batch);
        assert.ok(batch.pages.reduce((sum,p)=>sum+Buffer.byteLength(p.text,"utf8"),0)<=650000);
        for(const p of batch.pages){
            assert.equal(/[\uD800-\uDBFF]$/u.test(p.text),false);
            assert.equal(/^[\uDC00-\uDFFF]/u.test(p.text),false);
        }
        pages.push(...batch.pages);
        offset=batch.nextOffset;
        previousSha=batch.sourceSha256;
        if(++chunks>16)throw Error("invalid continuation");
    }while(offset!==null);
    assert.equal(assembleDocumentPages(pages).text,unicode);
});

test("small documents still support verified single chunk",()=>{
    const source="  ВВЕДЕНИЕ\nАвтор формулирует вывод.\n\n";
    const chunk=buildDocumentPageChunk(source,"short.txt");
    assert.equal(chunk.hasMore,false);
    assert.equal(chunk.sourceFullyCollected,true);
    assert.equal(chunk.pageCount,1);
    assert.equal(assembleDocumentPages(chunk.pages).text,source);
    assert.equal(buildDocumentPageChunk("", "empty.txt"),null);
    assert.equal(buildDocumentPageChunk(source, ""),null);
});
