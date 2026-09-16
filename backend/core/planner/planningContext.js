/*
 * =========================================================
 * JESSICA PLANNING CONTEXT
 * =========================================================
 *
 * Единый формат дополнительного контекста Planner.
 *
 *
 * Используется:
 *
 * Experience
 * Replanner
 * Learning
 * Earnings
 *
 *
 * Этот модуль НЕ:
 *
 * - ищет опыт;
 * - изменяет Skills;
 * - принимает решения;
 * - вызывает AI.
 *
 * Только нормализует данные.
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


        /*
         * Опыт Jessica
         */

        experience:
            null,



        /*
         * Глобальные правила
         */

        sourceRules:
            [],



        /*
         * Ограничения задачи
         */

        constraints:
            [],



        /*
         * Дополнительные инструкции Planner
         */

        instructions:
            [],



        /*
         * Подсказки для построения маршрута
         */

        plannerHints:
            [],



        /*
         * Служебные данные
         *
         * retry
         * execution
         * learning
         */

        metadata:
            {}

    };

}



/*
 * =========================================================
 * NORMALIZE ARRAY
 * =========================================================
 */


function normalizeArray(
    value
) {

    if (
        !Array.isArray(value)
    ) {

        return [];

    }


    return value

        .map(
            item =>
                String(
                    item || ""
                )
                .trim()
        )

        .filter(
            Boolean
        );

}



/*
 * =========================================================
 * NORMALIZE OBJECT
 * =========================================================
 */


function normalizeObject(
    value
) {

    if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ) {

        return {};

    }


    return {
        ...value
    };

}



/*
 * =========================================================
 * NORMALIZE EXPERIENCE
 * =========================================================
 */


function normalizeExperience(
    experience
) {


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return null;

    }



    return {

        ...experience,


        strategy:
            normalizeArray(
                experience.strategy
            ),


        sourcePriority:
            normalizeArray(
                experience.sourcePriority
            ),


        validationRules:
            normalizeArray(
                experience.validationRules
            ),


        failurePatterns:
            normalizeArray(
                experience.failurePatterns
            ),


        successfulPatterns:
            normalizeArray(
                experience.successfulPatterns
            ),


        avoidPatterns:
            normalizeArray(
                experience.avoidPatterns
            )

    };

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

            normalizeExperience(
                context.experience
            ),



        sourceRules:

            normalizeArray(
                context.sourceRules
            ),



        constraints:

            normalizeArray(
                context.constraints
            ),



        instructions:

            normalizeArray(
                context.instructions
            ),



        plannerHints:

            normalizeArray(
                context.plannerHints
            ),



        metadata:

            normalizeObject(
                context.metadata
            )


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



        normalized.plannerHints.length > 0 ||



        Object.keys(
            normalized.metadata
        ).length > 0


    );

}
