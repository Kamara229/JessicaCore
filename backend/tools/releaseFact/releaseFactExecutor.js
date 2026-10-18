import {
    normalizeText,
    normalizeTitle,
    normalizeUrl,
    getSubjectTokens
} from "./releaseFactText.js";

import {
    extractPreferredReleaseSection
} from "./releaseFactSection.js";

import {
    buildCandidates,
    selectCandidate
} from "./releaseFactCandidates.js";

import {
    buildEvidenceFragment,
    verifyEvidence
} from "./releaseFactEvidence.js";


export function executeReleaseFact(args) {
    const content = normalizeText(args?.content);
    const sourceUrl = normalizeUrl(args?.sourceUrl);
    const sourceTitle = normalizeTitle(args?.sourceTitle);
    const subject = normalizeText(args?.subject);

    if (!content) {
        return {
            success:false,
            needsClarification:true,
            text:"Для извлечения версии и даты требуется содержимое источника.",
            data:null
        };
    }

    const subjectTokens = getSubjectTokens(subject);
    const preferredSection = extractPreferredReleaseSection(content);

    let candidates = buildCandidates({
        section:preferredSection,
        sourceTitle,
        sourceUrl,
        subjectTokens
    });

    if (candidates.length === 0 && preferredSection.preferred === true) {
        candidates = buildCandidates({
            section:{
                text:content,
                start:0,
                preferred:false
            },
            sourceTitle,
            sourceUrl,
            subjectTokens
        });
    }

    const selected = selectCandidate(candidates);

    if (!selected || selected.score < 15) {
        return {
            success:false,
            text:"В источнике не удалось детерминированно подтвердить стабильную версию и дату выпуска именно запрошенного продукта.",
            data:{
                sourceUrl,
                sourceTitle,
                subjectTokens,
                releaseSectionFound:preferredSection.preferred === true,
                candidates:candidates.slice(0, 8).map(item => ({
                    version:item.version,
                    releaseDate:item.releaseDate,
                    score:item.score,
                    line:item.line
                }))
            }
        };
    }

    const evidenceFragment = buildEvidenceFragment(
        preferredSection.preferred === true
            ? preferredSection.text
            : content,
        selected
    );

    if (!verifyEvidence(evidenceFragment, selected)) {
        return {
            success:false,
            text:"Извлечённые версия и дата не подтверждаются одним фрагментом источника.",
            data:{
                sourceUrl,
                version:selected.version,
                releaseDate:selected.releaseDate
            }
        };
    }

    return {
        success:true,
        text:
            "Актуальная стабильная версия — "
            + selected.version
            + ". Дата выпуска — "
            + selected.releaseDate
            + "."
            + (sourceUrl ? " Источник: " + sourceUrl : ""),
        data:{
            subject,
            subjectTokens,
            version:selected.version,
            releaseDate:selected.releaseDate,
            sourceUrl,
            sourceTitle,
            evidenceVerified:true,
            evidenceFragment,
            releaseSectionFound:preferredSection.preferred === true,
            extractor:"release-fact-v2"
        }
    };
}
