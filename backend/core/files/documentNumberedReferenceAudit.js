import {createHash} from "node:crypto";
import {buildDocumentSectionIndex} from "./documentSectionIndex.js";
import {linkNumberedDocumentReferences} from "./documentNumberedReferenceLinks.js";

const sha=value=>createHash("sha256").update(value,"utf8").digest("hex");
const MAX_REFERENCES=600;
const BIB=/^(?:список\s+(?:использованных\s+)?(?:источников|литературы)|библиография|references)$/iu;
const NUMBERED_LINE=/^[ \t]{0,8}(?:\[(\d{1,4})\]|(\d{1,4})[.)])[ \t]+\S/u;
const PAGE_RE=/^(?:\d{1,4})(?:\s*,\s*(?:с|c|p|pp)\.?\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)?$/iu;

function parseBracketNumbers(candidate) {
    const parts=candidate.split(";").map(x=>x.trim());
    if(!parts.length||parts.length>6||parts.some(x=>!PAGE_RE.test(x)))
        return null;
    return parts.map(x=>Number(/^\d{1,4}/u.exec(x)[0]));
}

/**
 * Heuristic citation/bibliography navigation over checksum-verified text.
 * These are syntactic candidates, not proof that an author's sources exist,
 * are relevant, are correctly paginated or have been independently checked.
 */
export function auditDocumentNumberedReferences(source,expectedSha256) {
    if(typeof source!=="string"||!source.length||source.length>256000||
        !/^[a-f0-9]{64}$/u.test(String(expectedSha256||""))||
        sha(source)!==expectedSha256)return null;
    const index=buildDocumentSectionIndex(source);
    if(!index.complete||index.sourceChars!==source.length)return null;
    const bibliography=index.sections.filter(s=>BIB.test(s.title.trim()));
    const heading=bibliography.length===1?bibliography[0]:null;
    const bodyEnd=heading?.startOffset??source.length;
    const refs=[],bibEntries=[];
    let clipped=false;
    for(const match of source.slice(0,bodyEnd).matchAll(/\[([^\]\r\n]{1,145})\]/gu)){
        const numbers=parseBracketNumbers(match[1]);
        if(!numbers)continue;
        const position=match.index;
        const section=index.sections.find(s=>
            position>=s.startOffset&&position<s.endOffset);
        for(const number of numbers){
            if(refs.length>=MAX_REFERENCES){clipped=true;break;}
            refs.push({number,startOffset:position,
                endOffset:position+match[0].length,
                sectionIndex:section?.startOffset===undefined?null:
                    index.sections.indexOf(section),
                literalSha256:sha(match[0])});
        }
        if(clipped)break;
    }
    if(heading){
        const bibliographyText=source.slice(heading.startOffset,heading.endOffset);
        for(const match of bibliographyText.matchAll(/[^\r\n]+/gu)){
            const entry=NUMBERED_LINE.exec(match[0]);
            if(!entry)continue;
            if(bibEntries.length>=MAX_REFERENCES){clipped=true;break;}
            const number=Number(entry[1]||entry[2]);
            bibEntries.push({number,
                startOffset:heading.startOffset+match.index,
                endOffset:heading.startOffset+match.index+match[0].length,
                lineSha256:sha(match[0])});
        }
    }
    const citationNumbers=[...new Set(refs.map(x=>x.number))].sort((a,b)=>a-b);
    const bibliographyNumbers=[...new Set(bibEntries.map(x=>x.number))].sort((a,b)=>a-b);
    const refsSet=new Set(citationNumbers),bibliographySet=new Set(bibliographyNumbers);
    const duplicated=new Map();
    for(const item of bibEntries)duplicated.set(item.number,
        (duplicated.get(item.number)||0)+1);
    const referenceLinks=linkNumberedDocumentReferences(
        source,index,refs,bibEntries,Boolean(heading));
    if(!referenceLinks)return null;
    return {sourceSha256:expectedSha256,sourceChars:source.length,
        referenceLinks,
        bibliographyHeadingFound:Boolean(heading),
        multipleBibliographiesDetected:bibliography.length>1,
        citationMentions:refs.length,
        bibliographyEntries:bibEntries.length,
        inTextReferences:refs,
        bibliographyEntryLocations:bibEntries,
        missingEntryNumbers:heading?citationNumbers.filter(n=>!bibliographySet.has(n)):[],
        unreferencedEntryNumbers:heading?
            bibliographyNumbers.filter(n=>!refsSet.has(n)):[],
        duplicateEntryNumbers:[...duplicated].filter(([,n])=>n>1).map(([n])=>n),
        completeCandidateScan:!clipped,
        bibliographyCompletenessConfirmed:false,
        sourceRelevanceVerified:false,
        pageAccuracyVerified:false};
}
