/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL METADATA BUILDER
 * =========================================================
 *
 * Формирует provenance и служебные
 * данные Experience Skill.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText,
    normalizePositiveInteger
} from "./skillUtils.js";


export function buildSkillMetadata({

    proposedExperience,

    metadata = {}

} = {}) {


    const safeMetadata =

        isObject(
            metadata
        )

            ? metadata

            : {};


    return {


        proposalId:

            safeMetadata.proposalId

            ||

            null,



        queueItemId:

            safeMetadata.queueItemId

            ||

            null,



        traceId:

            safeMetadata.traceId

            ||

            null,



        learnedFrom:

            normalizeText(
                safeMetadata.learnedFrom
            )

            ||

            "learning_pipeline",



        createdBy:

            normalizeText(
                safeMetadata.createdBy
            )

            ||

            "jessica-learning",



        sourceExperience:

            safeMetadata.sourceExperience

            ||

            null,



        /*
         * Candidate provenance
         */


        candidateType:

            normalizeText(
                proposedExperience?.candidateType
            )

            ||

            normalizeText(
                safeMetadata.candidateType
            )

            ||

            null,



        improvementType:

            normalizeText(
                proposedExperience?.improvementType
            )

            ||

            normalizeText(
                safeMetadata.improvementType
            )

            ||

            null,



        baseVersion:

            normalizePositiveInteger(
                proposedExperience?.baseVersion
            )

            ||

            normalizePositiveInteger(
                safeMetadata.baseVersion
            )

            ||

            null,



        dynamicPattern:

            safeMetadata.dynamicPattern === true,



        /*
         * Старый metadata.learning
         * пока сохраняем как диагностический
         * Learning Context.
         *
         * Это НЕ канонические Skill Metrics.
         *
         * Канонические метрики:
         *
         * skill.learning
         */


        learning:

            safeMetadata.learning

            ??

            null

    };

}
