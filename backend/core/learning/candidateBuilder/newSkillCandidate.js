/*
 * =========================================================
 * JESSICA NEW SKILL CANDIDATE BUILDER
 * =========================================================
 *
 * Pattern
 *   +
 * Successful Execution Trace
 *        ↓
 * NEW_SKILL Candidate
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


    const examples =

        buildCandidateExamples(
            trace
        );


    const metrics =

        buildCandidateLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });


    const requiredTools =

        resolveNewSkillRequiredTools({

            pattern,

            trace

        });


    return {

        /*
         * =================================================
         * IDENTITY
         * =================================================
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
         * =================================================
         * KNOWLEDGE
         * =================================================
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


        requiredTools,


        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples,


        /*
         * =================================================
         * LEARNING METRICS
         * =================================================
         */


        ...metrics,


        /*
         * =================================================
         * META
         * =================================================
         */


        candidateType:

            "NEW_SKILL",


        source:

            "execution-trace"

    };

}
