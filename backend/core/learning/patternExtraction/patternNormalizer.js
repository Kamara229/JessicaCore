/*
 * =========================================================
 * JESSICA PATTERN NORMALIZER v1
 * =========================================================
 *
 * Приводит AI Pattern
 * к каноническому Experience Pattern.
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - принимает Learning Decision;
 * - сохраняет Experience.
 *
 * =========================================================
 */


import {
    buildLearningSkillId
} from "../../../experience/learning/learningSkillBuilder.js";





const MAX_WORKFLOW_STEPS =
    10;


const MAX_TRIGGER_PATTERNS =
    15;


const MAX_RULES =
    10;





function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeText(
    value,
    maxLength = 1000
) {

    return String(
        value || ""
    )
    .trim()
    .slice(
        0,
        maxLength
    );

}





function normalizeStringArray(
    value,
    limit = 10
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }



    const result = [];



    for(
        const item
        of value
    ){

        const normalized =

            normalizeText(
                item,
                500
            );



        if(
            !normalized
        ){

            continue;

        }



        if(
            result.includes(
                normalized
            )
        ){

            continue;

        }



        result.push(
            normalized
        );



        if(
            result.length >= limit
        ){

            break;

        }

    }



    return result;

}





/*
 * =========================================================
 * REQUIRED TOOLS
 * =========================================================
 */


function normalizeRequiredTools(
    requestedTools,
    usedTools
) {

    const requested =

        normalizeStringArray(
            requestedTools,
            20
        );



    const allowed =

        normalizeStringArray(
            usedTools,
            20
        );



    if(
        allowed.length === 0
    ){

        return [];

    }



    const allowedMap =

        new Map(

            allowed.map(

                item => [

                    item.toLowerCase(),

                    item

                ]

            )

        );



    const result = [];



    for(
        const item
        of requested
    ){

        const actual =

            allowedMap.get(
                item.toLowerCase()
            );



        if(
            actual &&
            !result.includes(actual)
        ){

            result.push(
                actual
            );

        }

    }



    return result;

}





/*
 * =========================================================
 * NORMALIZE PATTERN
 * =========================================================
 */


export function normalizeExtractedPattern({

    pattern,

    evidence

} = {}) {


    if(
        !isObject(
            pattern
        )
    ){

        return null;

    }



    const name =

        normalizeText(
            pattern.name,
            160
        );



    if(
        !name
    ){

        return null;

    }



    const workflow =

        normalizeStringArray(
            pattern.workflow,
            MAX_WORKFLOW_STEPS
        );



    const triggerPatterns =

        normalizeStringArray(
            pattern.triggerPatterns,
            MAX_TRIGGER_PATTERNS
        );



    const validationRules =

        normalizeStringArray(
            pattern.validationRules,
            MAX_RULES
        );



    const hasFailureEvidence =

        Array.isArray(
            evidence?.failures
        )

        &&

        evidence.failures.length > 0;



    return {


        id:

            buildLearningSkillId(
                name
            ),



        name,



        category:

            normalizeText(
                pattern.category,
                120
            )

            ||

            "general",



        description:

            normalizeText(
                pattern.description,
                1000
            ),



        triggerPatterns,



        keywords:

            triggerPatterns,



        workflow,



        validationRules,



        constraints:

            normalizeStringArray(
                pattern.constraints,
                MAX_RULES
            ),



        requiredTools:

            normalizeRequiredTools(

                pattern.requiredTools,

                evidence?.tools

            ),



        successfulPatterns:

            normalizeStringArray(
                pattern.successfulPatterns,
                MAX_RULES
            ),



        failurePatterns:

            hasFailureEvidence

                ? normalizeStringArray(
                    pattern.failurePatterns,
                    MAX_RULES
                )

                : [],



        avoidPatterns:

            hasFailureEvidence

                ? normalizeStringArray(
                    pattern.avoidPatterns,
                    MAX_RULES
                )

                : [],



        source:

            "ai-pattern-extraction",



        extractorVersion:

            "v2"

    };

}
