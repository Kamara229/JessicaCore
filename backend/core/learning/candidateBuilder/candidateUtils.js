/*
 * =========================================================
 * JESSICA CANDIDATE UTILS
 * =========================================================
 *
 * Общие нормализаторы и небольшие
 * pure helpers Candidate Builder.
 *
 * =========================================================
 */


/*
 * =========================================================
 * OBJECT
 * =========================================================
 */


export function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


/*
 * =========================================================
 * TEXT
 * =========================================================
 */


export function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


/*
 * =========================================================
 * NUMBER
 * =========================================================
 */


export function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


/*
 * =========================================================
 * UNIT
 * =========================================================
 */


export function normalizeUnit(
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
 * ARRAY
 * =========================================================
 */


export function normalizeArray(
    value
) {

    return Array.isArray(value)

        ? [
            ...value
        ]

        : [];

}


/*
 * =========================================================
 * STRING ARRAY
 * =========================================================
 */


export function normalizeStringArray(
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

            normalizeText(
                item
            );


        if(
            !text
        ){

            continue;

        }


        if(
            !result.includes(
                text
            )
        ){

            result.push(
                text
            );

        }

    }


    return result;

}


/*
 * =========================================================
 * MERGE STRING ARRAYS
 * =========================================================
 */


export function mergeStringArrays(
    ...arrays
) {

    return normalizeStringArray(

        arrays.flatMap(

            value =>

                Array.isArray(value)

                    ? value

                    : []

        )

    );

}


/*
 * =========================================================
 * OBSERVED TOOLS
 * =========================================================
 *
 * Канонический источник:
 *
 * trace.executedTools
 *
 * Это фактические Execution Tools,
 * а не Planner steps.
 *
 * =========================================================
 */


export function resolveObservedTools(
    trace
) {

    return normalizeStringArray(
        trace?.executedTools
    );

}


/*
 * =========================================================
 * TRACE SUCCESS
 * =========================================================
 */


export function resolveTraceSuccess(
    trace
) {

    if(
        typeof trace?.result?.success ===
        "boolean"
    ){

        return trace.result.success;

    }


    if(
        typeof trace?.success ===
        "boolean"
    ){

        return trace.success;

    }


    const completed =

        Number(

            trace?.statistics?.completed

            ??

            trace?.stats?.completed

            ??

            0

        );


    return (

        Number.isFinite(
            completed
        )

        &&

        completed > 0

    );

}


/*
 * =========================================================
 * SKILL DATA
 * =========================================================
 *
 * Поддерживает:
 *
 * Skill
 *
 * и transport wrapper:
 *
 * {
 *   experience: Skill
 * }
 *
 * =========================================================
 */


export function resolveSkillData(
    skill
) {

    if(
        !isObject(
            skill
        )
    ){

        return null;

    }


    if(
        isObject(
            skill.experience
        )
    ){

        return skill.experience;

    }


    return skill;

}
