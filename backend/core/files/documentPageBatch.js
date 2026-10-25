import {pageDocumentText} from "./documentTextWindow.js";
import {assembleDocumentPages} from "./documentPageAssembly.js";

const MAX_BATCH_PAGES=16;
/** Upload once, extract once, return a complete bounded UTF-16 page series. */
export function buildDocumentPageBatch(source,name) {
    if(typeof source!=="string"||!source.length||typeof name!=="string"||!name)
        return null;
    const pages=[];
    let offset=0;
    do {
        if(pages.length>=MAX_BATCH_PAGES) return null;
        const page=pageDocumentText(source,offset);
        if(!page?.text) return null;
        pages.push({...page,name});
        offset=page.nextOffset;
    }while(offset!==null);
    const reassembled=assembleDocumentPages(pages);
    return reassembled?.complete===true&&reassembled.text===source
        ? pages:null;
}
