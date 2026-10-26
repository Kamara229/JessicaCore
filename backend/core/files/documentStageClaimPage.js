import {checkStageReportOrder} from "./documentStageReportOrder.js";

const CLAIM=/^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): [^\r\n]+$/gmu;
const EVIDENCE=/^  Основание: [^\r\n]+$/gmu;
const REF=/\[О([1-9]\d{0,2}):Ф1:Ф[1-9]\d{0,2}\]/gu;
const PAGE_CLAIMS=12;
const MAX_PAGE_CHARS=16000;

/** Source-checked, server-attested claim snippets for LATER cross-stage synthesis.
 * Never silently omit claims or treat this page as an AI-synthesized answer.
 */
export function planAuthenticatedStageClaimPage({
    source,verified,reports,task,secret,claimOffset=0
}={}){
    if(!Number.isSafeInteger(claimOffset)||claimOffset<0)return null;
    const proof=checkStageReportOrder(source,verified,reports,task,secret);
    if(!proof?.allStageReportsReceived||!proof.reportOriginAuthenticated||
       !proof.reportBodySourceQuotesChecked||
       !proof.sectionCoverage?.allSectionsAuthenticated)return null;
    const claims=[];
    for(let index=0;index<reports.length;index++){
        const report=reports[index];
        const lines=[...report.text.matchAll(CLAIM)].map(x=>x[0]);
        const quotes=[...report.text.matchAll(EVIDENCE)].map(x=>x[0]);
        if(lines.length!==quotes.length||!lines.length)return null;
        for(let n=0;n<lines.length;n++){
            const refs=[...lines[n].matchAll(REF)].map(x=>Number(x[1]));
            if(!refs.length||refs.some(x=>x!==refs[0])||
               refs[0]<1||refs[0]>verified.totalWindows||
               refs[0]<=0)return null;
            const window=verified.windows[refs[0]-1];
            if(!window||window.number!==refs[0])return null;
            claims.push({stageNumber:index+1,windowNumber:refs[0],
                sectionIndex:window.sectionIndex,
                startOffset:window.startOffset,endOffset:window.endOffset,
                claim:lines[n],evidence:quotes[n]});
        }
    }
    if(claims.length!==proof.reportedGroundedClaims||
       claimOffset>claims.length)return null;
    let bytes=0;
    const page=[];
    for(let i=claimOffset;i<claims.length&&page.length<PAGE_CLAIMS;i++){
        const claim=claims[i];
        const next=JSON.stringify(claim);
        if(next.length>MAX_PAGE_CHARS)return null;
        if(page.length&&bytes+next.length>MAX_PAGE_CHARS)break;
        page.push(claim);bytes+=next.length;
    }
    if(claimOffset<claims.length&&!page.length)return null;
    const nextClaimOffset=claimOffset+page.length;
    return {sourceSha256:proof.sourceSha256,
        taskContextMatched:true,reportOriginAuthenticated:true,
        sourceQuotePresenceChecked:true,
        totalWindows:proof.totalWindows,
        totalClaims:claims.length,claimOffset,
        returnedClaims:page.length,nextClaimOffset:
            nextClaimOffset<claims.length?nextClaimOffset:null,
        allClaimsPaginated:nextClaimOffset===claims.length,
        claims:page,semanticSynthesisPerformed:false,
        completeSourceAnalysisConfirmed:false,
        scientificCorrectnessVerified:false};
}
