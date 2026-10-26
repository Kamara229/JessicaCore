import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
import {createHash} from "node:crypto";

const sha256=text=>createHash("sha256").update(text,"utf8").digest("hex");
const REF=/^(?:Д[1-3]:)?О(?:[1-9]|[12]\d|3[0-2]):Ф1:Ф[1-9]\d{0,2}$/u;
const KIND=new Set(["source_fact","author_claim","inference","verify_external"]);

/** Link each verified claim to structural provenance, not to a model heading. */
export function mapVerifiedClaimsToSourceSections(verifiedClaims,lineage){
    if(!Array.isArray(verifiedClaims)||!verifiedClaims.length||
       verifiedClaims.length>224||!lineage||
       !Array.isArray(lineage.citations)||!lineage.citations.length||
       !Number.isSafeInteger(lineage.citationCount)||
       lineage.citationCount!==lineage.citations.length||
       !Number.isSafeInteger(lineage.verifiedClaimCount)||
       verifiedClaims.length!==lineage.verifiedClaimCount)return null;
    const known=new Map();
    for(const entry of lineage.citations){
        if(!REF.test(String(entry?.ref||""))||known.has(entry.ref)||
           !Number.isSafeInteger(entry.fileIndex)||entry.fileIndex<1||entry.fileIndex>3||
           !Number.isSafeInteger(entry.windowNumber)||entry.windowNumber<1||
           !Number.isSafeInteger(entry.sectionIndex)||entry.sectionIndex<0||
           !Number.isSafeInteger(entry.rootSectionIndex)||entry.rootSectionIndex<0||
           !Array.isArray(entry.headingPath)||!entry.headingPath.length||
           !entry.headingPath.every(t=>typeof t==="string"&&t.length>0)||
           !Number.isSafeInteger(entry.exactQuoteCount)||entry.exactQuoteCount<0)
            return null;
        known.set(entry.ref,entry);
    }
    const claims=[];
    for(let i=0;i<verifiedClaims.length;i++){
        const finding=verifiedClaims[i];
        if(typeof finding?.text!=="string"||!finding.text.trim()||
           finding.text.length>800||!KIND.has(finding.kind)||
           !Number.isSafeInteger(finding.fileIndex)||finding.fileIndex<1||
           finding.fileIndex>3||!Number.isSafeInteger(finding.windowNumber)||
           finding.windowNumber<1||finding.windowNumber>MAX_VERIFIED_DOCUMENT_WINDOWS||
           !Array.isArray(finding.refs)||!finding.refs.length||
           finding.refs.length>3)return null;
        const refs=[...new Set(finding.refs)];
        const sources=refs.map(r=>known.get(r));
        if(refs.some(r=>!REF.test(r))||sources.some(s=>!s))return null;
        const first=sources[0];
        if(sources.some(s=>s.fileIndex!==finding.fileIndex||
           s.fileIndex!==first.fileIndex||
           s.windowNumber!==finding.windowNumber||
           s.sectionIndex!==first.sectionIndex||
           s.rootSectionIndex!==first.rootSectionIndex||
           JSON.stringify(s.headingPath)!==JSON.stringify(first.headingPath)))
            return null;
        const exactQuoteRefs=refs.filter(r=>known.get(r).exactQuoteCount>0);
        // The digest records the statement identity without duplicating the
        // user's claim text in diagnostic metadata or reusable Skills.
        const statementSha256=sha256(JSON.stringify([
            finding.kind,finding.text.trim(),refs
        ]));
        claims.push({index:i+1,statementSha256,kind:finding.kind,
            fileIndex:finding.fileIndex,
            windowNumber:finding.windowNumber,
            sectionIndex:first.sectionIndex,
            rootSectionIndex:first.rootSectionIndex,
            headingPath:[...first.headingPath],
            evidenceRefs:refs,
            exactQuoteRefs,
            allRefsHaveUniqueVerbatimQuote:exactQuoteRefs.length===refs.length});
    }
    return {count:claims.length,claims,sourceHeadingsHeuristic:true,
        logicalInferenceVerified:false,independentSourcesVerified:false};
}

/** Combine only per-file validated records; no private document contents. */
export function mergeClaimProvenanceReports(reports){
    if(!Array.isArray(reports)||reports.length<2||reports.length>3)return null;
    const claims=[];
    for(let i=0;i<reports.length;i++){
        const file=reports[i];
        if(file?.fileIndex!==i+1||!file.provenance||
           !Array.isArray(file.provenance.claims)||
           file.provenance.count!==file.provenance.claims.length||
           file.provenance.count<1)return null;
        for(const claim of file.provenance.claims){
            if(claim.fileIndex!==file.fileIndex||
               !Array.isArray(claim.evidenceRefs)||
               !claim.evidenceRefs.length||
               claim.evidenceRefs.some(r=>!r.startsWith(`Д${file.fileIndex}:`)))
                return null;
            claims.push({...claim,index:claims.length+1});
        }
    }
    if(claims.length>224)return null;
    return {count:claims.length,claims,sourceHeadingsHeuristic:true,
        logicalInferenceVerified:false,independentSourcesVerified:false};
}
