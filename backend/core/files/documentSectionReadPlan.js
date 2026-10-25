import {buildDocumentSectionIndex} from "./documentSectionIndex.js";
import {selectDocumentSectionPage} from "./documentSectionWindow.js";

/** A bounded, contiguous reading plan, NOT evidence that the model read it. */
export function planDocumentSectionWindows(source) {
    const index=buildDocumentSectionIndex(source);
    const state={sourceChars:typeof source==="string"?source.length:0,windows:[]};
    if(typeof source!=="string"||!source||!index.complete||!index.sections.length)
        return {...state,complete:false,reason:"index-incomplete"};
    let offset=0;
    for(let i=0;i<index.sections.length;i++){
        const section=index.sections[i];
        if(section.startOffset!==offset||section.endOffset<=offset)
            return {...state,complete:false,reason:"section-gap"};
        let cursor=offset;
        while(cursor<section.endOffset){
            if(state.windows.length>=16)
                return {...state,complete:false,reason:"window-limit"};
            const page=selectDocumentSectionPage(source,index,i,cursor);
            if(!page?.text||page.offset!==cursor||page.text.length>16000)
                return {...state,complete:false,reason:"invalid-window"};
            const end=cursor+page.text.length;
            if(end>section.endOffset||end<=cursor)
                return {...state,complete:false,reason:"invalid-range"};
            state.windows.push({number:state.windows.length+1,
                sectionIndex:i,title:section.title,startOffset:cursor,
                endOffset:end,text:page.text});
            cursor=end;
        }
        offset=section.endOffset;
    }
    return {...state,complete:offset===source.length,
        reason:offset===source.length?null:"not-covered"};
}
