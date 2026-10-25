import {assembleDocumentPages} from "./documentPageAssembly.js";
import {createLosslessWindowEvidence} from "./documentWindowEvidence.js";
import {planDocumentSectionWindows} from "./documentSectionReadPlan.js";
import {validateAndRenderDocumentAnswer} from "./documentAnswerContract.js";
import {identifyDocumentAnswer} from "./documentResponseIdentity.js";
import {beginWindowLedger,noteVerifiedWindow,windowLedgerSummary} from "./documentWindowLedger.js";
import {locateCitedDocumentFragments} from "./documentCitationSpans.js";
import {createVerifiedClaimRecords,compareVerifiedDocumentClaims} from "./documentClaimComparison.js";
import {auditDocumentSections} from "./documentSectionCoverage.js";
import {locateVerbatimQuoteRanges} from "./documentVerbatimQuotes.js";
import {markDocumentQuotaInterruption} from "./documentQuotaCoverage.js";
import {measureDocumentCitationDensity} from "./documentEvidenceDensity.js";
import {buildDocumentEvidenceLineage} from "./documentEvidenceLineage.js";
import {mapVerifiedClaimsToSourceSections} from "./documentClaimProvenance.js";
import {auditDocumentNumberedReferences} from "./documentNumberedReferenceAudit.js";
import {summarizeSectionReferenceCoverage} from "./documentSectionReferenceCoverage.js";

/** Use only cryptographically verified complete pages; fail closed on gaps. */
export async function analyzePagedLongDocument({task,attachment,generate,embedReceipt=true}) {
    const name=String(attachment?.name||"Документ");
    const pages=attachment?.pages;
    const hash=pages?.[0]?.sourceSha256;
    const valid=typeof hash==="string"&&/^[a-f\d]{64}$/u.test(hash)&&
        (!pages[0].name||pages[0].name===name);
    const assembled=valid?assembleDocumentPages(pages):null;
    if(!assembled)return {success:false,mode:"document",stage:"document-pages",
        text:"Страницы документа не прошли проверку целостности."};
    const plan=planDocumentSectionWindows(assembled.text);
    if(!plan.complete)return {success:false,mode:"document",stage:"document-read-plan",
        text:"Полный анализ требует продолжения по частям. Не подтверждена полнота.",
        readingCoverage:{plannedWindows:plan.windows.length,processedWindows:0,
            complete:false,reason:plan.reason,sourceChars:assembled.text.length}};
    const ledger=beginWindowLedger(plan,hash);
    if(!ledger)return {success:false,mode:"document",stage:"document-source-coverage",
        text:"Не совпадает контрольная сумма окон документа."};
    let processed=0,kept=0,rejected=0;
    const answers=[],citationSpans=[],verifiedClaims=[];
    const quoteRanges=[];
    const quoteCoverage={evidenceQuotes:0,exactLocatedQuotes:0,ambiguousQuotes:0,
        nonLiteralQuotes:0,allVerbatimQuoteOffsetsLocated:false};
    for(const window of plan.windows){
        const excerpt={name,text:window.text,truncated:false};
        const bundle=createLosslessWindowEvidence(excerpt);
        if(!bundle?.exactSourceCoverage)return {success:false,mode:"document",
            stage:"document-window-coverage",
            text:"Невозможно подтвердить передачу полного текста окна.",
            readingCoverage:{...windowLedgerSummary(ledger),complete:false}};
        const prompt=[
            "ЗАДАНИЕ:",String(task).trim().slice(0,3000),
            "СОДЕРЖИМОЕ ТОЛЬКО ОДНОГО ОКНА:",
            `Окно ${window.number}, раздел «${window.title}», символы ${window.startOffset}–${window.endOffset}.`,
            bundle.promptText,
            "Верни JSON с проверяемыми цитатами. Не заявляй, что анализировал другие окна."
        ].join("\n");
        let checked;
        try {
            checked=validateAndRenderDocumentAnswer(
                await generate(prompt,task,[excerpt]),bundle,task);
        }catch(error){
            if(error?.status===429)
                throw markDocumentQuotaInterruption(error,
                    windowLedgerSummary(ledger),window.number);
            return {success:false,mode:"document",stage:"document-window-error",
                text:"Окно не обработано; полный анализ не завершён.",
                readingCoverage:{...windowLedgerSummary(ledger),complete:false}};
        }
        if(!checked.success)return {success:false,mode:"document",stage:"document-window-evidence",
            text:"Для одного из окон не подтверждены цитаты. Полный анализ не завершён.",
            readingCoverage:{...windowLedgerSummary(ledger),complete:false}};
        const verifiedSpans=locateCitedDocumentFragments(window,bundle,checked);
        const verifiedRecords=createVerifiedClaimRecords(
            checked.verifiedFindings,verifiedSpans,window.number);
        if(!verifiedRecords)return {success:false,mode:"document",stage:"document-claim-links",
            text:"Невозможно сопоставить проверенные выводы с источниками.",
            readingCoverage:windowLedgerSummary(ledger)};
        if(!verifiedSpans)return {success:false,mode:"document",stage:"document-citation-offsets",
            text:"Источник цитаты не совпал с контрольными смещениями.",
            readingCoverage:windowLedgerSummary(ledger)};
        const exactQuotes=locateVerbatimQuoteRanges(window,bundle,checked);
        if(!exactQuotes)return {success:false,mode:"document",stage:"document-quote-check",
            text:"Не удалось подтвердить происхождение цитат.",
            readingCoverage:windowLedgerSummary(ledger)};
        if(!noteVerifiedWindow(ledger,window,checked))
            return {success:false,mode:"document",stage:"document-window-ledger",
                text:"Ответ относится не к ожидаемому фрагменту.",
                readingCoverage:windowLedgerSummary(ledger)};
        processed++;kept+=checked.kept;rejected+=checked.rejected;
        const scoped=checked.text.replace(/\[Ф1:Ф(\d+)\]/gu,
            (_whole,n)=>`[О${window.number}:Ф1:Ф${n}]`);
        answers.push(`Окно ${window.number} — ${window.title} (${window.startOffset}–${window.endOffset})\n${scoped}`);
        citationSpans.push(...verifiedSpans);
        verifiedClaims.push(...verifiedRecords);
        quoteRanges.push(...exactQuotes.ranges);
        for(const key of ["evidenceQuotes","exactLocatedQuotes",
            "ambiguousQuotes","nonLiteralQuotes"])quoteCoverage[key]+=exactQuotes[key];
    }
    const body=answers.join("\n\n")+
        "\n\nГраницы проверки: все окна переданы для анализа и цитаты сверены. " +
        "Это не доказывает научную или юридическую правильность выводов.";
    const claimComparison=compareVerifiedDocumentClaims(verifiedClaims,citationSpans);
    if(!claimComparison)return {success:false,mode:"document",stage:"document-comparison",
        text:"Не удалось проверить межраздельные ссылки на источники.",
        readingCoverage:windowLedgerSummary(ledger)};
    const sectionCoverage=auditDocumentSections(plan,ledger,citationSpans,verifiedClaims);
    if(!sectionCoverage?.allSectionsProcessed)
        return {success:false,mode:"document",stage:"document-section-coverage",
            text:"Один или несколько разделов не подтверждены.",
            readingCoverage:windowLedgerSummary(ledger)};
    quoteCoverage.allVerbatimQuoteOffsetsLocated=
        quoteCoverage.evidenceQuotes>0 &&
        quoteCoverage.evidenceQuotes===quoteCoverage.exactLocatedQuotes;
    const evidenceDensity=measureDocumentCitationDensity(plan,quoteRanges);
    if(!evidenceDensity)return {success:false,mode:"document",stage:"document-evidence-density",
        text:"Не удалось проверить привязку цитат к прочитанным фрагментам.",
        readingCoverage:windowLedgerSummary(ledger)};
    const evidenceLineage=buildDocumentEvidenceLineage(
        sectionCoverage,windowLedgerSummary(ledger),citationSpans,verifiedClaims,quoteRanges);
    if(!evidenceLineage)return {success:false,mode:"document",stage:"document-evidence-lineage",
        text:"Нарушена связь цитат с разделами документа.",
        readingCoverage:windowLedgerSummary(ledger)};
    const claimProvenance=mapVerifiedClaimsToSourceSections(verifiedClaims,evidenceLineage);
    if(!claimProvenance)return {success:false,mode:"document",stage:"document-claim-provenance",
        text:"Не удалось установить исходные разделы для выводов.",
        readingCoverage:windowLedgerSummary(ledger)};
    const referenceAudit=auditDocumentNumberedReferences(assembled.text,hash);
    if(!referenceAudit)return {success:false,mode:"document",
        stage:"document-reference-audit",
        text:"Не удалось проверить нумерованные источники по исходному тексту.",
        readingCoverage:windowLedgerSummary(ledger)};
    const referenceSectionCoverage=summarizeSectionReferenceCoverage(
        referenceAudit,sectionCoverage.hierarchy);
    if(!referenceSectionCoverage)return {success:false,mode:"document",
        stage:"document-reference-section-coverage",
        text:"Не удалось проверить нумерованные источники по разделам.",
        readingCoverage:windowLedgerSummary(ledger)};
    const audit=embedReceipt?identifyDocumentAnswer(body,kept,rejected):null;
    return {success:true,mode:"document",validated:false,evidenceChecked:true,
        groundedClaims:kept,unsupportedClaimsDropped:rejected,
        readingCoverage:windowLedgerSummary(ledger),
        citationSpans,verifiedClaims,claimComparison,sectionCoverage,
        quoteRanges,quoteCoverage,
        evidenceDensity,
        evidenceLineage,
        claimProvenance,
        referenceAudit,referenceSectionCoverage,
        responseId:audit?.id||null,responseSha256:audit?.sha256||null,
        text:audit?.text||body};
}
