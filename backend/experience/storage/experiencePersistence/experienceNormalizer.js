/*
 * =========================================================
 * JESSICA EXPERIENCE NORMALIZER v1
 * =========================================================
 *
 * Каноническая нормализация
 * Experience Skill.
 *
 *
 * НЕ:
 *
 * - читает Supabase;
 * - сохраняет Skill;
 * - вычисляет версии;
 * - принимает Learning Decision.
 *
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value &&
        typeof value === "object" &&
        !Array.isArray(value)

    );

}


export function normalizeExperienceText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


export function normalizeExperienceVersion(
    value
) {

    const version =
        Number(value);


    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return null;

    }


    return version;

}


export function normalizeExperienceConfidence(
    value
) {

    const number =
        Number(value);


    if(
        !Number.isFinite(number)
    ){

        return 0;

    }


    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}


export function normalizeExperienceStringArray(
    value
) {

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    const result = [];


    for(
        const item
        of value
    ){

        const text =

            normalizeExperienceText(
                item
            );


        if(
            text &&
            !result.includes(text)
        ){

            result.push(
                text
            );

        }

    }


    return result;

}


export function normalizeExperienceObjectArray(
    value
) {

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value

        .filter(
            isObject
        )

        .map(
            item => ({
                ...item
            })
        );

}


function normalizeLearning(
    value
) {

    if(
        !isObject(value)
    ){

        return {

            occurrences:
                0,

            successCount:
                0,

            failureCount:
                0,

            successRate:
                0,

            maturity:
                0,

            maturityLevel:
                null,

            confidence:
                0

        };

    }


    return {

        ...value,

        occurrences:

            Math.max(
                Number(value.occurrences || 0),
                0
            ),

        successCount:

            Math.max(
                Number(value.successCount || 0),
                0
            ),

        failureCount:

            Math.max(
                Number(value.failureCount || 0),
                0
            ),

        successRate:

            normalizeExperienceConfidence(
                value.successRate
            ),

        maturity:

            normalizeExperienceConfidence(
                value.maturity
            ),

        maturityLevel:

            normalizeExperienceText(
                value.maturityLevel
            )

            ||

            null,

        confidence:

            normalizeExperienceConfidence(
                value.confidence
            )

    };

}


function normalizeStatistics(
    value
) {

    if(
        !isObject(value)
    ){

        return {

            successfulRuns:
                0,

            failedRuns:
                0,

            lastUsedAt:
                null,

            lastResult:
                null

        };

    }


    return {

        successfulRuns:

            Math.max(
                Number(
                    value.successfulRuns || 0
                ),
                0
            ),

        failedRuns:

            Math.max(
                Number(
                    value.failedRuns || 0
                ),
                0
            ),

        lastUsedAt:

            value.lastUsedAt

            ||

            null,

        lastResult:

            value.lastResult

            ??

            null

    };

}


/*
 * =========================================================
 * NORMALIZE EXPERIENCE
 * =========================================================
 */


export function normalizeExperienceSkill(
    experience
) {

    if(
        !isObject(
            experience
        )
    ){

        return null;

    }


    const id =

        normalizeExperienceText(

            experience.id

            ||

            experience.skillId

        );


    const version =

        normalizeExperienceVersion(
            experience.version
        );


    const previousVersion =

        experience.previousVersion === null
        ||
        experience.previousVersion === undefined

            ? null

            : normalizeExperienceVersion(
                experience.previousVersion
            );


    return {

        ...experience,


        /*
         * IDENTITY
         */


        id,


        name:

            normalizeExperienceText(
                experience.name
            ),


        description:

            normalizeExperienceText(
                experience.description
            ),


        category:

            normalizeExperienceText(
                experience.category
            )

            ||

            "general",


        /*
         * VERSION
         */


        version,


        previousVersion,


        mode:

            normalizeExperienceText(
                experience.mode
            )

            ||

            "create",


        /*
         * STATUS
         */


        enabled:

            experience.enabled !== false,


        confidence:

            normalizeExperienceConfidence(

                experience.learning?.confidence

                ??

                experience.confidence

            ),


        /*
         * KNOWLEDGE
         */


        workflow:

            normalizeExperienceStringArray(
                experience.workflow
            ),


        triggerPatterns:

            normalizeExperienceStringArray(
                experience.triggerPatterns
            ),


        keywords:

            normalizeExperienceStringArray(
                experience.keywords
            ),


        tags:

            normalizeExperienceStringArray(
                experience.tags
            ),


        validationRules:

            normalizeExperienceStringArray(
                experience.validationRules
            ),


        constraints:

            normalizeExperienceStringArray(
                experience.constraints
            ),


        strategy:

            normalizeExperienceStringArray(
                experience.strategy
            ),


        sourcePriority:

            normalizeExperienceStringArray(
                experience.sourcePriority
            ),


        requiredTools:

            normalizeExperienceStringArray(
                experience.requiredTools
            ),


        successfulPatterns:

            normalizeExperienceStringArray(
                experience.successfulPatterns
            ),


        failurePatterns:

            normalizeExperienceStringArray(
                experience.failurePatterns
            ),


        avoidPatterns:

            normalizeExperienceStringArray(
                experience.avoidPatterns
            ),


        examples:

            normalizeExperienceObjectArray(
                experience.examples
            ),


        /*
         * LEARNING MEMORY
         */


        learning:

            normalizeLearning(
                experience.learning
            ),


        /*
         * RUNTIME MEMORY
         */


        statistics:

            normalizeStatistics(
                experience.statistics
            ),


        /*
         * META
         */


        metadata:

            isObject(
                experience.metadata
            )

                ? {
                    ...experience.metadata
                }

                : {}

    };

}
