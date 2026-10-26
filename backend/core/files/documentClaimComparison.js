import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
import {createHash} from "node:crypto";

const ALLOWED=new Set(["source_fact","author_claim","inference","verify_external"]);
const fingerprint=text=>createHash("sha256").update(text,"utf8").digest("hex");
const canonical=text=>text.normalize("NFKC").replace(/\s+/gu," ").trim().toLocaleLowerCase("ru");
const REF=/^(?:Д[1-3]:)?О(?:[1-9]|[12]\d|3[0-2]):Ф1:Ф\d{1,3}$/u;

/** Each claim originates in a quote-validated response, not raw AI output. */
export function createVerifiedClaimRecords(findings,spans,windowNumber,fileIndex=1) {
    if(!Array.isArray(findings)||!findings.length||findings.length>14||
       !Array.isArray(spans)||!spans.length||
       !Number.isSafeInteger(windowNumber)||windowNumber<1||windowNumber>MAX_VERIFIED_DOCUMENT_WINDOWS||
       !Number.isSafeInteger(fileIndex)||fileIndex<1||fileIndex>3)return null;
    const available=new Set(spans.map(s=>s?.ref));
    const records=[];
    for(const finding of findings){
        if(typeof finding?.text!=="string"||!finding.text.trim()||
           finding.text.length>800||!ALLOWED.has(finding.kind)||
           !Array.isArray(finding.refs)||!finding.refs.length||finding.refs.length>3)
           return null;
        const refs=[...new Set(finding.refs.map(ref=>`О${windowNumber}:${ref}`))];
        if(refs.some(ref=>!REF.test(ref)||!available.has(ref)))return null;
        records.push({kind:finding.kind,text:finding.text.trim(),refs,
            fileIndex,windowNumber,heading:String(finding.heading||"").slice(0,90)});
    }
    return records;
}

/** Adjust already-verified local references when merging multiple files. */
export function scopeVerifiedClaimRecords(records,fileIndex,scopedSpans) {
    if(!Array.isArray(records)||!records.length||
       !Number.isSafeInteger(fileIndex)||fileIndex<1||fileIndex>3||
       !Array.isArray(scopedSpans))return null;
    const known=new Set(scopedSpans.map(s=>s?.ref));
    const scoped=[];
    for(const record of records){
        if(!Array.isArray(record?.refs)||!record.refs.length)return null;
        const refs=record.refs.map(ref=>`Д${fileIndex}:${ref}`);
        if(refs.some(ref=>!REF.test(ref)||!known.has(ref)))return null;
        scoped.push({...record,fileIndex,refs});
    }
    return scoped;
}

/**
 * Compare identical normalized formulations, not legal truth or paraphrase meaning.
 * Repetition across files is not independent corroboration.
 */
export function compareVerifiedDocumentClaims(records,spans) {
    if(!Array.isArray(records)||!records.length||records.length>224||
       !Array.isArray(spans)||!spans.length)return null;
    const known=new Set(spans.map(s=>s?.ref));
    const groups=new Map();
    for(const record of records){
        if(typeof record?.text!=="string"||!record.text.trim()||
           record.text.length>800||!ALLOWED.has(record.kind)||
           !Number.isSafeInteger(record.fileIndex)||record.fileIndex<1||record.fileIndex>3||
           !Number.isSafeInteger(record.windowNumber)||record.windowNumber<1||record.windowNumber>MAX_VERIFIED_DOCUMENT_WINDOWS||
           !Array.isArray(record.refs)||!record.refs.length||
           record.refs.some(ref=>typeof ref!=="string"||!REF.test(ref)||!known.has(ref)))return null;
        const normalized=canonical(record.text);
        if(!normalized)return null;
        const key=record.kind+"\u0000"+normalized;
        if(!groups.has(key))groups.set(key,{
            claimSha256:fingerprint(key),kind:record.kind,
            text:record.text.slice(0,220),occurrences:[]});
        const group=groups.get(key);
        group.occurrences.push({fileIndex:record.fileIndex,
            windowNumber:record.windowNumber,refs:[...new Set(record.refs)]});
    }
    const repeats=[...groups.values()].filter(g=>
        new Set(g.occurrences.map(o=>o.fileIndex+":"+o.windowNumber)).size>1);
    const crossFile=repeats.filter(g=>
        new Set(g.occurrences.map(o=>o.fileIndex)).size>1);
    return {totalVerifiedClaims:records.length,distinctFormulations:groups.size,
        repeatedFormulations:repeats.length,crossFileExactRepeats:crossFile.length,
        examples:repeats.slice(0,8),semanticContradictionsAssessed:false,
        independentCorroborationEstablished:false};
}
