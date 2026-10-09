/*
 * =========================================================
 * JESSICA NEW SKILL CANDIDATE BUILDER v2
 * =========================================================
 *
 * Pattern
 *   +
 * Successful Execution Evidence
 *        ↓
 * NEW_SKILL Candidate
 *
 *
 * requiredTools формируются не из
 * одного Execution, а из накопленного
 * подтверждённого Tool Evidence.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeArray,
    normalizeText,
    normalizeStringArray
} from "./candidateUtils.js";


import {
    buildCandidateExamples
} from "./candidateExamples.js";


import {
    buildCandidateLearningMetrics
} from "./candidateMetrics.js";


import {
    resolveNewSkillRequiredTools
} from "./candidateToolKnowledge.js";


/*
 * =========================================================
 * BUILD
 * =========================================================
 */


export function buildNewSkillCandidate({

    pattern,

    trace,

    confidence,

    maturity,

    occurrences

}) {

    if(
        !isObject(
            pattern
        )
    ){

        return null;

    }


    const skillId =

        normalizeText(

            pattern.id

            ||

            pattern.skillId

        );


    const name =

        normalizeText(
            pattern.name
        );


    if(
        !skillId
        ||
        !name
    ){

        return null;

    }


    /*
     * =====================================================
     * EVIDENCE
     * =====================================================
     */


    const examples =

        buildCandidateExamples(
            trace
        );


    /*
     * =====================================================
     * METRICS
     * =====================================================
     */


    const metrics =

        buildCandidateLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });


    /*
     * =====================================================
     * TOOL KNOWLEDGE
     * =====================================================
     *
     * Один Execution ещё не превращает
     * Tool в requiredTools.
     *
     * Candidate Memory позже объединит
     * независимые Examples и повторно
     * рассчитает requiredTools.
     *
     * =====================================================
     */


    const toolKnowledge =

        resolveNewSkillRequiredTools({

            examples

        });


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        /*
         * IDENTITY
         */


        skillId,


        name,


        category:

            normalizeText(
                pattern.category
            )

            ||

            "general",


        description:

            normalizeText(
                pattern.description
            ),


        /*
         * KNOWLEDGE
         */


        workflow:

            normalizeArray(
                pattern.workflow
            ),


        validationRules:

            normalizeStringArray(
                pattern.validationRules
            ),


        triggerPatterns:

            normalizeStringArray(

                pattern.triggerPatterns

                ||

                pattern.keywords

            ),


        keywords:

            normalizeStringArray(
                pattern.keywords
            ),


        tags:

            normalizeStringArray(
                pattern.tags
            ),


        constraints:

            normalizeStringArray(
                pattern.constraints
            ),


        strategy:

            normalizeStringArray(
                pattern.strategy
            ),


        sourcePriority:

            normalizeStringArray(
                pattern.sourcePriority
            ),


        requiredTools:

            toolKnowledge.requiredTools,


        successfulPatterns:

            normalizeStringArray(
                pattern.successfulPatterns
            ),


        failurePatterns:

            normalizeStringArray(
                pattern.failurePatterns
            ),


        avoidPatterns:

            normalizeStringArray(
                pattern.avoidPatterns
            ),


        /*
         * EVIDENCE
         */


        examples,


        /*
         * LEARNING METRICS
         */


        ...metrics,


        /*
         * META
         */


        candidateType:

            "NEW_SKILL",


        source:

            "execution-trace"

    };

}
