import {createHash} from "node:crypto";
import {buildVerifiedSectionHierarchy} from "./documentSectionHierarchy.js";

const sha=text=>createHash("sha256").update(text,"utf8").digest("hex");

/** Audits exactly which indexed source sections received citation-validated responses.
 * Section headings are heuristic, and passing this audit does not prove truth.
 */
export function auditDocumentSections(plan,ledger,spans,claims){
    if(plan?.complete!==true||!Array.isArray(plan.windows)||!plan.windows.length||
       plan.windows.length>16||!ledger||!Array.isArray(ledger.expected)||
       !Array.isArray(ledger.confirmed)||ledger.expected.length!==plan.windows.length||
       !Array.isArray(spans)||!Array.isArray(claims))return null;
    const windows=new Map(), sections=new Map();
    let end=0;
    for(let i=0;i<plan.windows.length;i++){
        const w=plan.windows[i],expected=ledger.expected[i];
        if(w?.number!==i+1||expected?.number!==w.number||
           w.sectionIndex!==expected.sectionIndex||
           !Number.isSafeInteger(w.startOffset)||w.startOffset!==end||
           !Number.isSafeInteger(w.endOffset)||w.endOffset<=end||
           w.endOffset!==expected.endOffset||w.startOffset!==expected.startOffset||
           typeof w.text!=="string"||sha(w.text)!==expected.windowSha256)return null;
        end=w.endOffset;
        windows.set(w.number,w);
        const title=String(w.title||"Документ").slice(0,180);
        if(!sections.has(w.sectionIndex)){
            sections.set(w.sectionIndex,{sectionIndex:w.sectionIndex,
                title,startOffset:w.startOffset,endOffset:w.startOffset,
                sourceChars:0,plannedWindows:0,processedWindows:0,
                citedFragments:0,verifiedClaims:0,complete:false});
        }
        const current=sections.get(w.sectionIndex);
        if(current.endOffset!==w.startOffset||current.title!==title)return null;
        current.endOffset=w.endOffset;
        current.sourceChars+=w.endOffset-w.startOffset;
        current.plannedWindows++;
    }
    if(end!==plan.sourceChars||ledger.sourceChars!==end||
       ledger.confirmed.length>ledger.expected.length)return null;
    for(let i=0;i<ledger.confirmed.length;i++){
        const row=ledger.confirmed[i],expected=ledger.expected[i];
        if(row?.windowSha256!==expected.windowSha256||
           row?.number!==expected.number||
           row?.sectionIndex!==expected.sectionIndex||
           row?.startOffset!==expected.startOffset||
           row?.endOffset!==expected.endOffset||
           !Number.isSafeInteger(row.groundedClaims)||row.groundedClaims<1||
           !sections.has(row.sectionIndex))return null;
        sections.get(row.sectionIndex).processedWindows++;
    }
    const seen=new Set();
    for(const span of spans){
        const win=windows.get(span?.windowNumber);
        if(!win||span.sectionIndex!==win.sectionIndex||
           !Number.isSafeInteger(span.startOffset)||
           !Number.isSafeInteger(span.endOffset)||
           span.startOffset<win.startOffset||span.endOffset>win.endOffset||
           span.endOffset<=span.startOffset||
           sha(win.text.slice(span.startOffset-win.startOffset,
              span.endOffset-win.startOffset))!==span.fragmentSha256||
           typeof span.ref!=="string"||
           !new RegExp(`^О${win.number}:Ф1:Ф[1-9]\\d*$`,"u").test(span.ref)||
           seen.has(span.ref)||span.windowNumber>ledger.confirmed.length)return null;
        seen.add(span.ref);
        sections.get(win.sectionIndex).citedFragments++;
    }
    for(const claim of claims){
        const win=windows.get(claim?.windowNumber);
        if(!win||claim.fileIndex!==1||
           !Array.isArray(claim.refs)||!claim.refs.length||
           claim.refs.some(ref=>!seen.has(ref))||
           claim.windowNumber>ledger.confirmed.length)return null;
        sections.get(win.sectionIndex).verifiedClaims++;
    }
    const result=[...sections.values()];
    for(const item of result)item.complete=item.processedWindows===item.plannedWindows;
    const hierarchy=buildVerifiedSectionHierarchy(plan,result);
    if(!hierarchy)return null;
    return {sections:result,hierarchy,sectionCount:result.length,
        completedSections:result.filter(s=>s.complete).length,
        allSectionsProcessed:result.every(s=>s.complete),
        allSectionsCited:result.every(s=>s.citedFragments>0),
        totalSourceChars:end,sectionHeadingsHeuristic:true,
        semanticCorrectnessConfirmed:false};
}
