import {pageDocumentText} from "./documentTextWindow.js";

const MAX_CHUNK_PAGES=16;
const MAX_CHUNK_TEXT_BYTES=650000;
const SHA256=/^[0-9a-f]{64}$/u;

/**
 * Stateless bounded retrieval for files larger than one complete extraction.
 * The client reuploads the SAME original file with nextOffset and digest;
 * only assembleDocumentPages(all pages) can certify full-source collection.
 */
export function buildDocumentPageChunk(source,name,offset=0,expectedSha256=null){
    if(typeof source!=="string"||!source.length||
       typeof name!=="string"||!name||name.length>100||
       !Number.isSafeInteger(offset)||offset<0||offset>=source.length||
       (expectedSha256!==null&&!SHA256.test(String(expectedSha256))))
        return null;
    const pages=[];
    let cursor=offset,usedBytes=0,sourceSha256=null;
    for(let i=0;i<MAX_CHUNK_PAGES;i++){
        const page=pageDocumentText(source,cursor);
        if(!page?.text||page.offset!==cursor||
           page.extractedChars!==source.length)return null;
        if(sourceSha256===null)sourceSha256=page.sourceSha256;
        if(page.sourceSha256!==sourceSha256||
           (expectedSha256!==null&&expectedSha256!==sourceSha256))
            return null;
        const pageBytes=Buffer.byteLength(page.text,"utf8");
        if(pageBytes>MAX_CHUNK_TEXT_BYTES||(!i?false:usedBytes+pageBytes>MAX_CHUNK_TEXT_BYTES))
            break;
        pages.push({...page,name});
        usedBytes+=pageBytes;
        if(!page.hasMore)break;
        if(!Number.isSafeInteger(page.nextOffset)||page.nextOffset<=cursor)
            return null;
        cursor=page.nextOffset;
    }
    if(!pages.length)return null;
    const last=pages.at(-1);
    const endOffset=last.offset+last.text.length;
    const hasMore=endOffset<source.length;
    if(hasMore!==last.hasMore||
       (hasMore&&last.nextOffset!==endOffset)||
       (!hasMore&&last.nextOffset!==null))return null;
    return {pages,startOffset:offset,endOffset,
        hasMore,nextOffset:hasMore?endOffset:null,
        pageCount:pages.length,extractedChars:source.length,
        sourceSha256,sourceFullyCollected:offset===0&&!hasMore,
        modelAnalysisPerformed:false,
        semanticCompletenessConfirmed:false};
}
