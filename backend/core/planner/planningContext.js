/*
 * =========================================================
 * JESSICA PLANNING CONTEXT
 * =========================================================
 *
 * Формирует дополнительный контекст,
 * который может быть передан Planner.
 *
 * В будущем сюда будут поступать:
 *
 * - опыт Jessica;
 * - успешные алгоритмы;
 * - правила источников;
 * - ограничения задания;
 * - контекст Earnings;
 * - пользовательские инструкции.
 *
 *
 * ВАЖНО:
 *
 * Этот модуль НЕ решает задачу.
 *
 * Он только приводит дополнительный
 * контекст к единому формату.
 *
 * =========================================================
 */


/*
 * =========================================================
 * EMPTY CONTEXT
 * =========================================================
 */


export function createEmptyPlanningContext() {

    return {

        experience:
            null,

        sourceRules:
            [],

        constraints:
            [],

        instructions:
            [],

        metadata:
            {}

    };

}


/*
 * =========================================================
 * NORMALIZE STRING ARRAY
 * =========================================================
 */


function normalizeStringArray(
    value
) {

    if (!Array.isArray(value)) {

        return [];

    }


    return value
        .map(
            item =>
                String(
                    item || ""
                ).trim()
        )
        .filter(
            Boolean
        );

}


/*
 * =========================================================
 * NORMALIZE CONTEXT
 * =========================================================
 */


export function normalizePlanningContext(
    context
) {

    if (
        !context ||
        typeof context !== "object"
    ) {

        return createEmptyPlanningContext();

    }


    return {

        experience:
            context.experience &&
            typeof context.experience === "object"
                ? context.experience
                : null,


        sourceRules:
            normalizeStringArray(
                context.sourceRules
            ),


        constraints:
            normalizeStringArray(
                context.constraints
            ),


        instructions:
            normalizeStringArray(
                context.instructions
            ),


        metadata:
            context.metadata &&
            typeof context.metadata === "object"
                ? {
                    ...context.metadata
                }
                : {}

    };

}


/*
 * =========================================================
 * HAS CONTEXT
 * =========================================================
 */


export function hasPlanningContext(
    context
) {

    const normalized =
        normalizePlanningContext(
            context
        );


    return Boolean(

        normalized.experience ||

        normalized.sourceRules.length > 0 ||

        normalized.constraints.length > 0 ||

        normalized.instructions.length > 0 ||

        Object.keys(
            normalized.metadata
        ).length > 0

    );

}
