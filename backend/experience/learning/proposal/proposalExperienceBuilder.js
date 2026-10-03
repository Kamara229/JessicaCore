/*
 * =========================================================
 * JESSICA PROPOSED EXPERIENCE BUILDER
 * =========================================================
 *
 * Learning Candidate
 *        ↓
 * Proposed Experience
 *
 *
 * Proposed Experience является
 * полным предполагаемым состоянием
 * будущего Experience Skill.
 *
 *
 * Используется:
 *
 * NEW_SKILL
 *      ↓
 * Skill v1
 *
 * SKILL_IMPROVEMENT
 *      ↓
 * Skill vNext
 *
 *
 * НЕ:
 *
 * - принимает Approval;
 * - сохраняет Skill;
 * - создаёт Version;
 * - работает с Supabase.
 *
 * =========================================================
 */


import {
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger,
    normalizeStringArray,
    normalizeObjectArray
} from "./proposalUtils.js";


/*
 * =========================================================
 * LEARNING METRICS
 * =========================================================
 */


function buildLearningMetrics(
    candidate,
    confidence
) {

    const examples =

        normalizeObjectArray(
            candidate?.examples
        );


    const calculatedSuccessCount =

        examples.filter(

            item =>
                item?.success === true

        )
        .length;


    const calculatedFailureCount =

        examples.filter(

            item =>
                item?.success === false

        )
        .length;



    /*
     * ВАЖНО:
     *
     * 0 является допустимым значением.
     * Поэтому здесь нельзя использовать:
     *
     * candidate.successCount || calculated
     */


    const hasSuccessCount =

        Number.isFinite(

            Number(
                candidate?.successCount
            )

        );


    const hasFailureCount =

        Number.isFinite(

            Number(
                candidate?.failureCount
            )

        );


    const successCount =

        hasSuccessCount

            ? Math.max(
                Number(
                    candidate.successCount
                ),
                0
            )

            : calculatedSuccessCount;


    const failureCount =

        hasFailureCount

            ? Math.max(
                Number(
                    candidate.failureCount
                ),
                0
            )

            : calculatedFailureCount;


    const occurrences =

        Math.max(

            normalizePositiveInteger(
                candidate?.occurrences
            ),

            examples.length,

            1

        );


    let successRate = 0;


    if(
        candidate?.successRate !== undefined
        &&
        candidate?.successRate !== null
    ){

        successRate =

            normalizeUnit(
                candidate.successRate
            );

    }else if(
        examples.length > 0
    ){

        successRate =

            Number(

                (
                    calculatedSuccessCount /
                    examples.length
                )
                .toFixed(2)

            );

    }


    return {

        occurrences,


        successCount,


        failureCount,


        successRate,


        maturity:

            normalizeUnit(
                candidate?.maturity
            ),


        maturityLevel:

            normalizeText(
                candidate?.maturityLevel
            )

            ||

            null,


        confidence:

            normalizeUnit(
                confidence
            )

    };

}


/*
 * =========================================================
 * BUILD EXPERIENCE
 * =========================================================
 */


export function buildProposedExperience({

    candidate,

    targetSkill,

    confidence

}) {


    const learning =

        buildLearningMetrics(

            candidate,

            confidence

        );


    return {


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        id:

            targetSkill.id,



        name:

            normalizeText(
                candidate.name
            ),



        description:

            normalizeText(
                candidate.description
            ),



        category:

            normalizeText(
                candidate.category
            )

            ||

            "general",



        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        workflow:

            normalizeStringArray(
                candidate.workflow
            ),



        triggerPatterns:

            normalizeStringArray(
                candidate.triggerPatterns
            ),



        keywords:

            normalizeStringArray(
                candidate.keywords
            ),



        tags:

            normalizeStringArray(
                candidate.tags
            ),



        validationRules:

            normalizeStringArray(
                candidate.validationRules
            ),



        constraints:

            normalizeStringArray(
                candidate.constraints
            ),



        strategy:

            normalizeStringArray(
                candidate.strategy
            ),



        sourcePriority:

            normalizeStringArray(
                candidate.sourcePriority
            ),



        requiredTools:

            normalizeStringArray(
                candidate.requiredTools
            ),



        successfulPatterns:

            normalizeStringArray(
                candidate.successfulPatterns
            ),



        failurePatterns:

            normalizeStringArray(
                candidate.failurePatterns
            ),



        avoidPatterns:

            normalizeStringArray(
                candidate.avoidPatterns
            ),



        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples:

            normalizeObjectArray(
                candidate.examples
            ),



        /*
         * =================================================
         * LEARNING METRICS
         * =================================================
         *
         * Верхние поля временно сохраняем
         * для совместимости текущего
         * Autonomy Policy.
         *
         * learning является новым
         * каноническим контейнером.
         *
         * =================================================
         */


        occurrences:

            learning.occurrences,



        successCount:

            learning.successCount,



        failureCount:

            learning.failureCount,



        successRate:

            learning.successRate,



        maturity:

            learning.maturity,



        maturityLevel:

            learning.maturityLevel,



        confidence:

            learning.confidence,



        learning,



        /*
         * =================================================
         * CANDIDATE META
         * =================================================
         */


        candidateType:

            normalizeText(
                candidate.candidateType
            )

            ||

            null,



        improvementType:

            normalizeText(
                candidate.improvementType
            )

            ||

            null,



        baseVersion:

            normalizePositiveInteger(
                candidate.baseVersion
            )

            ||

            targetSkill.version

            ||

            null,



        source:

            normalizeText(
                candidate.source
            )

            ||

            "execution-learning"

    };

}
