import {auditStageClaimFragmentLinks} from "./documentStageClaimCitationAudit.js";

/**
 * Compare stage-report body against its exact source window boundaries.
 * This checks source quotations, NOT the authenticity of an AI response.
 */
const MAX_REPORT_CHARS=350000;
const MAX_WINDOWS=16;
const CITATION=/\[О(\d{1,3}):Ф1:Ф([1-9]\d{0,2})\]/gu;
const EVIDENCE=/^  Основание: (.+)$/gmu;
const QUOTED=/«([^»]{1,260})»/gu;

const norm=s=>String(s).normalize("NFKC")
    .replace(/[\u00a0\s]+/gu," ")
    .replace(/[«»„“”]/gu,'"')
    .replace(/[‐‑‒–—−]/gu,"-")
    .trim().toLocaleLowerCase("ru");

export function inspectDocumentStageReportBody(report,stage,original){
    if(!report||typeof report.text!=="string"||
       report.text.length>MAX_REPORT_CHARS||
       !stage||!Array.isArray(stage.windows)||
       !stage.windows.length||stage.windows.length>MAX_WINDOWS||
       typeof original!=="string"||!original.length||
       !Number.isSafeInteger(stage.stageNumber)||
       !Number.isSafeInteger(stage.stageCount))
        return null;
    const prefix="Проверенный этап "+stage.stageNumber+
        " из "+stage.stageCount+".\n\n";
    const suffix="\n\nПроверены цитаты этапа. Полнота всего документа и достоверность выводов не подтверждены.";
    if(!report.text.startsWith(prefix)||!report.text.endsWith(suffix))
        return null;
    const middle=report.text.slice(prefix.length,-suffix.length);
    let cursor=0,bodyClaims=0,bodyQuotes=0,bodyRefs=0;
    for(let i=0;i<stage.windows.length;i++){
        const window=stage.windows[i];
        const heading="Окно "+window.number+" ("+window.startOffset+
            "–"+window.endOffset+"):\n";
        if(!middle.startsWith(heading,cursor)||
           window.startOffset<0||
           window.endOffset>original.length||
           window.endOffset<=window.startOffset)
            return null;
        const begin=cursor+heading.length;
        const after=i+1<stage.windows.length?stage.windows[i+1]:null;
        const next=after?"\n\nОкно "+after.number+" ("+after.startOffset+
            "–"+after.endOffset+"):\n":null;
        const end=next?middle.indexOf(next,begin):middle.length;
        if(end<=begin)return null;
        const body=middle.slice(begin,end);
        const citationMatches=[...body.matchAll(CITATION)];
        if(!citationMatches.length||citationMatches.length>60||
           citationMatches.some(m=>Number(m[1])!==window.number))
            return null;
        const claims=[...body.matchAll(/^• /gmu)];
        const validClaimLabels=[...body.matchAll(
            /^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): /gmu)];
        if(validClaimLabels.length!==claims.length)return null;
        const evidenceLines=[...body.matchAll(EVIDENCE)];
        if(!claims.length||claims.length>14||
           evidenceLines.length!==claims.length)
            return null;
        const source=norm(original.slice(window.startOffset,window.endOffset));
        let quotes=0;
        for(const [,line] of evidenceLines){
            const found=[...line.matchAll(QUOTED)];
            if(!found.length||found.length>42)return null;
            for(const match of found){
                const quote=norm(match[1]);
                if(quote.length<6||!source.includes(quote))
                    return null;
                quotes++;
            }
        }
        const linkAudit=auditStageClaimFragmentLinks(body,
            original.slice(window.startOffset,window.endOffset),window.number);
        if(!linkAudit||linkAudit.claims!==claims.length||
           linkAudit.quotes!==quotes||
           linkAudit.refs!==citationMatches.length)return null;
        const expected=report.readingCoverage?.verifiedWindows?.[i];
        if(!expected||expected.windowNumber!==window.number||
           expected.groundedClaims!==claims.length||
           expected.evidenceQuotes!==quotes)
            return null;
        bodyClaims+=claims.length;
        bodyQuotes+=quotes;
        bodyRefs+=citationMatches.length;
        cursor=next?end+2:middle.length;
    }
    if(cursor!==middle.length)return null;
    return {bodyClaims,bodyQuotes,bodyRefs,
        windowBodiesChecked:stage.windows.length,
        sourceQuotePresenceVerified:true,
        externalSourceTruthVerified:false,reportOriginAuthenticated:false};
}
