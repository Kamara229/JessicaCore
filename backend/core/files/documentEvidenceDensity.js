import {createHash} from "node:crypto";
import {createLosslessWindowEvidence} from "./documentWindowEvidence.js";

const hash=text=>createHash("sha256").update(text,"utf8").digest("hex");

/**
 * Separates submitted text from literal quote anchoring.
 * A cited 40-character phrase does NOT certify an entire 900-character chunk.
 * Summary is diagnostic, not a claim of source truth or semantic completeness.
 */
export function measureDocumentCitationDensity(plan,quoteRanges) {
    if(plan?.complete!==true||!Array.isArray(plan.windows)||
       !plan.windows.length||plan.windows.length>16||
       !Array.isArray(quoteRanges))return null;
    const byWindow=new Map(),sections=new Map();
    let previousEnd=0,totalChunks=0,citedChunks=0,quotedCharacters=0;
    for(const window of plan.windows){
        if(!Number.isSafeInteger(window?.number)||window.number!==byWindow.size+1||
           !Number.isSafeInteger(window.sectionIndex)||window.sectionIndex<0||
           typeof window.text!=="string"||!window.text.length||
           window.startOffset!==previousEnd||
           window.endOffset!==previousEnd+window.text.length)return null;
        const evidence=createLosslessWindowEvidence({name:"Документ",text:window.text});
        if(!evidence?.exactSourceCoverage)return null;
        const pieces=evidence.fragments.map(f=>({
            start:window.startOffset+f.startOffset,
            end:window.startOffset+f.endOffset,
            ref:`О${window.number}:${f.ref}`,
            quoted:[]
        }));
        byWindow.set(window.number,{window,parts:pieces});
        if(!sections.has(window.sectionIndex))sections.set(window.sectionIndex,{
            sectionIndex:window.sectionIndex,title:String(window.title||"Документ").slice(0,180),
            submittedCharacters:0,fragments:0,fragmentsWithExactQuotes:0,
            fragmentsWithoutExactQuotes:0,exactQuoteCharacters:0});
        const section=sections.get(window.sectionIndex);
        section.submittedCharacters+=window.text.length;
        section.fragments+=pieces.length;
        totalChunks+=pieces.length;
        previousEnd=window.endOffset;
    }
    if(previousEnd!==plan.sourceChars)return null;
    const seen=new Set();
    for(const quote of quoteRanges){
        const entry=byWindow.get(quote?.windowNumber);
        if(!entry||quote.sectionIndex!==entry.window.sectionIndex||
           !Number.isSafeInteger(quote.startOffset)||
           !Number.isSafeInteger(quote.endOffset)||
           quote.endOffset<=quote.startOffset||
           !/^[a-f0-9]{64}$/u.test(String(quote.quoteSha256||"")))return null;
        const piece=entry.parts.find(f=>f.ref===quote.ref &&
            quote.startOffset>=f.start && quote.endOffset<=f.end);
        if(!piece)return null;
        const original=entry.window.text.slice(
            quote.startOffset-entry.window.startOffset,
            quote.endOffset-entry.window.startOffset);
        if(hash(original)!==quote.quoteSha256)return null;
        const unique=quote.ref+":"+quote.startOffset+":"+quote.endOffset;
        if(seen.has(unique))return null;
        seen.add(unique);
        piece.quoted.push([quote.startOffset,quote.endOffset]);
    }
    for(const {window,parts} of byWindow.values()){
        const section=sections.get(window.sectionIndex);
        for(const fragment of parts){
            if(!fragment.quoted.length){section.fragmentsWithoutExactQuotes++;continue;}
            citedChunks++;
            section.fragmentsWithExactQuotes++;
            // The union avoids counting duplicate/overlapping quote characters.
            const intervals=fragment.quoted.sort((a,b)=>a[0]-b[0]);
            let currentStart=intervals[0][0],currentEnd=intervals[0][1];
            let sum=0;
            for(const [start,end] of intervals.slice(1)){
                if(start<=currentEnd) currentEnd=Math.max(currentEnd,end);
                else {sum+=currentEnd-currentStart;currentStart=start;currentEnd=end;}
            }
            sum+=currentEnd-currentStart;
            section.exactQuoteCharacters+=sum;
            quotedCharacters+=sum;
        }
    }
    return {submittedCharacters:previousEnd,fragments:totalChunks,
        fragmentsWithExactQuotes:citedChunks,
        fragmentsWithoutExactQuotes:totalChunks-citedChunks,
        exactQuoteCharacters:quotedCharacters,
        fractionFragmentsAnchored:totalChunks?citedChunks/totalChunks:0,
        sections:[...sections.values()],
        sourceFullySubmitted:true,
        allFragmentsLiterallyCited:citedChunks===totalChunks,
        semanticCompletenessEstablished:false};
}

/** Aggregate only verified, nonoverlapping identities; never combine private text. */
export function summarizeMultiDocumentCitationDensity(reports) {
    if(!Array.isArray(reports)||reports.length<2||reports.length>3)return null;
    let submittedCharacters=0,fragments=0,fragmentsWithExactQuotes=0;
    let fragmentsWithoutExactQuotes=0,exactQuoteCharacters=0;
    const files=[];
    for(let i=0;i<reports.length;i++){
        const item=reports[i];
        if(!item||item.fileIndex!==i+1||!item.report||
           !Number.isSafeInteger(item.report.fragments)||
           item.report.fragments<1||
           item.report.fragmentsWithExactQuotes+item.report.fragmentsWithoutExactQuotes!==item.report.fragments||
           item.report.sourceFullySubmitted!==true)return null;
        for(const key of ["submittedCharacters","fragments","fragmentsWithExactQuotes",
            "fragmentsWithoutExactQuotes","exactQuoteCharacters"]){
            if(!Number.isSafeInteger(item.report[key])||item.report[key]<0)return null;
        }
        submittedCharacters+=item.report.submittedCharacters;
        fragments+=item.report.fragments;
        fragmentsWithExactQuotes+=item.report.fragmentsWithExactQuotes;
        fragmentsWithoutExactQuotes+=item.report.fragmentsWithoutExactQuotes;
        exactQuoteCharacters+=item.report.exactQuoteCharacters;
        files.push({fileIndex:item.fileIndex,...item.report});
    }
    return {submittedCharacters,fragments,fragmentsWithExactQuotes,
        fragmentsWithoutExactQuotes,exactQuoteCharacters,
        fractionFragmentsAnchored:fragments?fragmentsWithExactQuotes/fragments:0,
        sourceFullySubmitted:true,
        allFragmentsLiterallyCited:fragmentsWithoutExactQuotes===0,
        semanticCompletenessEstablished:false,files};
}
