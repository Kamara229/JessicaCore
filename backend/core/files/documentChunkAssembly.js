import {assembleDocumentPages} from "./documentPageAssembly.js";

const MAX_CHUNKS=16;
const MAX_PAGES=256;
const MAX_CHARS=4_000_000;
const MAX_CHUNK_BYTES=650000;
const SHA256=/^[a-f0-9]{64}$/u;

/**
 * Verify stateless document-extraction continuations before passing any
 * extracted text to an AI. A partial series cannot be treated as a document.
 * Source integrity becomes true only after final SHA-256 reassembly.
 */
export function assembleDocumentChunks(chunks) {
    if(!Array.isArray(chunks)||!chunks.length||chunks.length>MAX_CHUNKS)
        return null;
    const first=chunks[0];
    const hash=first?.sourceSha256;
    const total=first?.extractedChars;
    const name=first?.pages?.[0]?.name;
    if(!SHA256.test(String(hash||""))||
       !Number.isSafeInteger(total)||total<1||total>MAX_CHARS||
       typeof name!=="string"||!name||name.length>100)
        return null;
    let next=0;
    const pages=[];
    for(let i=0;i<chunks.length;i++){
        const chunk=chunks[i];
        if(!chunk||!Array.isArray(chunk.pages)||
           !chunk.pages.length||chunk.pages.length>16||
           chunk.pageCount!==chunk.pages.length||
           chunk.startOffset!==next||
           chunk.extractedChars!==total||
           chunk.sourceSha256!==hash||
           chunk.sourceFullyCollected!==(i===0 && chunk.hasMore===false)||
           chunk.modelAnalysisPerformed!==false||
           chunk.semanticCompletenessConfirmed!==false||
           !Number.isSafeInteger(chunk.endOffset)||
           chunk.endOffset<=next||chunk.endOffset>total||
           chunk.hasMore!==(chunk.endOffset<total)||
           chunk.nextOffset!==(chunk.hasMore?chunk.endOffset:null)||
           (!chunk.hasMore&&i!==chunks.length-1))
            return null;
        let bytes=0;
        for(const page of chunk.pages){
            if(!page||typeof page.text!=="string"||!page.text.length||
               page.text.length>16000||page.name!==name||
               page.sourceSha256!==hash||page.extractedChars!==total||
               page.offset!==next||page.hasMore!==(next+page.text.length<total)||
               page.nextOffset!==(page.hasMore?next+page.text.length:null))
                return null;
            bytes+=Buffer.byteLength(page.text,"utf8");
            if(bytes>MAX_CHUNK_BYTES||pages.length>=MAX_PAGES)
                return null;
            pages.push(page);
            next+=page.text.length;
        }
        if(next!==chunk.endOffset)return null;
    }
    const complete=next===total;
    if(!complete) {
        return {complete:false,sourceIntegrityVerified:false,
            sourceSha256:hash,extractedChars:total,collectedChars:next,
            nextOffset:next,chunkCount:chunks.length,pageCount:pages.length,
            modelAnalysisPerformed:false,semanticCompletenessConfirmed:false};
    }
    const verified=assembleDocumentPages(pages);
    if(!verified?.complete||verified.text.length!==total)return null;
    return {complete:true,sourceIntegrityVerified:true,
        sourceSha256:hash,extractedChars:total,collectedChars:total,
        nextOffset:null,chunkCount:chunks.length,pageCount:pages.length,
        name,pages,text:verified.text,modelAnalysisPerformed:false,
        semanticCompletenessConfirmed:false};
}