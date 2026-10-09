/*
 * =========================================================
 * JESSICA CANDIDATE MERGE v2
 * =========================================================
 *
 * Existing Candidate
 *        +
 * Incoming Candidate
 *        ↓
 * Evidence Merge
 *        ↓
 * Metrics
 *        ↓
 * Tool Evidence Aggregation
 *        ↓
 * Candidate Memory
 *
 *
 * Важно:
 *
 * requiredTools больше НЕ объединяются
 * через union.
 *
 *
 * Вместо:
 *
 * web_search
 * +
 * web_search + web_fetch
 *
 *        ↓
 *
 * web_search + web_fetch
 *
 *
 * используем подтверждённое пересечение:
 *
 * web_search
 * ∩
 * web_search + web_fetch
 *
 *        ↓
 *
 * web_search
 *
 * =========================================================
 */


import {
    calculateExperienceConfidence,
    calculateExperienceMaturity,
    resolveExperienceMaturityLevel
} from "../experienceConfidence.js";


import {
    inferRequiredToolsFromExamples
} from "../candidateBuilder/candidateToolKnowledge.js";


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


function clampUnit(
    value
) {

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}


/*
 * =========================================================
 * STRING ARRAYS
 * =========================================================
 */


function mergeStringArrays(
    ...arrays
) {

    const result = [];


    for(
        const values
        of arrays
    ){

        if(
            !Array.isArray(values)
        ){

            continue;

        }


        for(
            const value
            of values
        ){

            const text =

                String(
                    value || ""
                )
                .trim();


            if(
                text
                &&
                !result.includes(
                    text
                )
            ){

                result.push(
                    text
                );

            }

        }

    }


    return result;

}


/*
 * =========================================================
 * EXAMPLE KEY
 * =========================================================
 */


function buildExampleKey(
    example
) {

    if(
        !example
        ||
        typeof example !== "object"
    ){

        return "";

    }


    if(
        example.traceId
    ){

        return (
            "trace:"
            +
            String(
                example.traceId
            )
        );

    }


    if(
        example
            ?.result
            ?.executionMeta
            ?.traceId
    ){

        return (
            "execution-trace:"
            +
            String(
                example.result.executionMeta.traceId
            )
        );

    }


    let resultText = "";


    try {


        resultText =

            JSON.stringify(
                example.result ?? null
            );


    }catch(error){


        resultText =

            String(
                example.result || ""
            );

    }


    return (

        String(
            example.task || ""
        )
        .trim()

        +

        "::"

        +

        resultText

    );

}


/*
 * =========================================================
 * MERGE EXAMPLES
 * =========================================================
 */


function mergeExamples(
    existingExamples,
    incomingExamples
) {

    const result = [];

    const keys =
        new Set();


    let added = 0;


    const push = (
        example,
        incoming = false
    ) => {

        if(
            !example
            ||
            typeof example !== "object"
        ){

            return;

        }


        const key =

            buildExampleKey(
                example
            );


        if(
            key
            &&
            keys.has(
                key
            )
        ){

            return;

        }


        if(
            key
        ){

            keys.add(
                key
            );

        }


        result.push({

            ...example

        });


        if(
            incoming
        ){

            added += 1;

        }

    };


    for(
        const example
        of (
            Array.isArray(existingExamples)

                ? existingExamples

                : []
        )
    ){

        push(
            example,
            false
        );

    }


    for(
        const example
        of (
            Array.isArray(incomingExamples)

                ? incomingExamples

                : []
        )
    ){

        push(
            example,
            true
        );

    }


    return {

        examples:
            result,

        added

    };

}


/*
 * =========================================================
 * SUCCESS METRICS
 * =========================================================
 */


function calculateExampleMetrics(
    examples
) {

    const safeExamples =

        Array.isArray(examples)

            ? examples

            : [];


    const successCount =

        safeExamples.filter(

            item =>
                item?.success === true

        )
        .length;


    const failureCount =

        safeExamples.filter(

            item =>
                item?.success === false

        )
        .length;


    const successRate =

        safeExamples.length > 0

            ? Number(

                (
                    successCount
                    /
                    safeExamples.length
                )
                .toFixed(2)

            )

            : 0;


    return {

        successCount,

        failureCount,

        successRate

    };

}


/*
 * =========================================================
 * INFER MATCH SCORE
 * =========================================================
 */


function inferMatchScore(
    candidate
) {

    if(
        candidate?.matchScore !== undefined
        &&
        candidate?.matchScore !== null
    ){

        return clampUnit(
            candidate.matchScore
        );

    }


    const confidence =

        clampUnit(
            candidate?.confidence
        );


    const successRate =

        clampUnit(
            candidate?.successRate
        );


    const occurrences =

        Math.max(

            normalizeNumber(
                candidate?.occurrences
            ),

            1

        );


    const repeatScore =

        Math.min(
            occurrences / 5,
            1
        );


    const estimated =

        (
            confidence
            -
            successRate * 0.4
            -
            repeatScore * 0.3
        )

        /

        0.3;


    return clampUnit(
        estimated
    );

}


/*
 * =========================================================
 * MERGE
 * =========================================================
 */


export function mergeLearningCandidates({

    existing,

    incoming,

    exactMatch = false,

    similarity = 0

} = {}) {

    if(
        !existing
        ||
        typeof existing !== "object"
    ){

        return incoming;

    }


    if(
        !incoming
        ||
        typeof incoming !== "object"
    ){

        return existing;

    }


    /*
     * =====================================================
     * EXAMPLES
     * =====================================================
     */


    const mergedExamples =

        mergeExamples(

            existing.examples,

            incoming.examples

        );


    /*
     * =====================================================
     * OCCURRENCES
     * =====================================================
     */


    const previousOccurrences =

        Math.max(

            normalizeNumber(
                existing.occurrences
            ),

            Array.isArray(
                existing.examples
            )

                ? existing.examples.length

                : 0,

            1

        );


    const occurrences =

        Math.max(

            previousOccurrences
            +
            mergedExamples.added,

            mergedExamples.examples.length,

            1

        );


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    const exampleMetrics =

        calculateExampleMetrics(
            mergedExamples.examples
        );


    /*
     * =====================================================
     * MATCH
     * =====================================================
     */


    const existingMatchScore =

        inferMatchScore(
            existing
        );


    const incomingMatchScore =

        inferMatchScore(
            incoming
        );


    const matchScore =

        exactMatch

            ? 1

            : Math.max(

                existingMatchScore,

                incomingMatchScore,

                clampUnit(
                    similarity
                )

            );


    /*
     * =====================================================
     * MATURITY + CONFIDENCE
     * =====================================================
     */


    const maturity =

        calculateExperienceMaturity(
            occurrences
        );


    const confidence =

        calculateExperienceConfidence({

            matchScore,

            successRate:
                exampleMetrics.successRate,

            occurrences

        });


    /*
     * =====================================================
     * TOOL KNOWLEDGE
     * =====================================================
     *
     * Пересчитывается из полного
     * накопленного Evidence.
     *
     * НЕ:
     *
     * merge(existing.requiredTools,
     *       incoming.requiredTools)
     *
     * =====================================================
     */


    const toolKnowledge =

        inferRequiredToolsFromExamples(
            mergedExamples.examples
        );


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        /*
         * STABLE IDENTITY
         */


        skillId:

            existing.skillId

            ||

            incoming.skillId

            ||

            null,


        name:

            existing.name

            ||

            incoming.name

            ||

            "",


        category:

            existing.category

            ||

            incoming.category

            ||

            "general",


        description:

            existing.description

            ||

            incoming.description

            ||

            "",


        /*
         * KNOWLEDGE
         */


        workflow:

            Array.isArray(
                existing.workflow
            )
            &&
            existing.workflow.length > 0

                ? [
                    ...existing.workflow
                ]

                : (
                    Array.isArray(
                        incoming.workflow
                    )

                        ? [
                            ...incoming.workflow
                        ]

                        : []
                ),


        triggerPatterns:

            mergeStringArrays(

                existing.triggerPatterns,

                incoming.triggerPatterns

            ),


        keywords:

            mergeStringArrays(

                existing.keywords,

                incoming.keywords

            ),


        tags:

            mergeStringArrays(

                existing.tags,

                incoming.tags

            ),


        validationRules:

            mergeStringArrays(

                existing.validationRules,

                incoming.validationRules

            ),


        constraints:

            mergeStringArrays(

                existing.constraints,

                incoming.constraints

            ),


        strategy:

            mergeStringArrays(

                existing.strategy,

                incoming.strategy

            ),


        sourcePriority:

            mergeStringArrays(

                existing.sourcePriority,

                incoming.sourcePriority

            ),


        /*
         * КРИТИЧНО:
         *
         * requiredTools выводится
         * из фактического Evidence.
         */


        requiredTools:

            toolKnowledge.requiredTools,


        successfulPatterns:

            mergeStringArrays(

                existing.successfulPatterns,

                incoming.successfulPatterns

            ),


        failurePatterns:

            mergeStringArrays(

                existing.failurePatterns,

                incoming.failurePatterns

            ),


        avoidPatterns:

            mergeStringArrays(

                existing.avoidPatterns,

                incoming.avoidPatterns

            ),


        /*
         * EVIDENCE
         */


        examples:

            mergedExamples.examples,


        /*
         * METRICS
         */


        occurrences,


        successCount:

            exampleMetrics.successCount,


        failureCount:

            exampleMetrics.failureCount,


        successRate:

            exampleMetrics.successRate,


        maturity,


        maturityLevel:

            resolveExperienceMaturityLevel(
                occurrences
            ),


        matchScore,


        confidence,


        /*
         * META
         */


        candidateType:

            "NEW_SKILL",


        source:

            existing.source

            ||

            incoming.source

            ||

            "candidate-memory"

    };

}
