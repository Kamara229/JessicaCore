/*
 * =========================================================
 * JESSICA PLANNING CONTEXT
 * =========================================================
 *
 * Единый формат контекста Jessica Planner.
 *
 *
 * Используется:
 *
 * Experience
 * Planner
 * Replanner
 * Learning
 * Earnings
 *
 *
 * Flow:
 *
 * External Context
 *        ↓
 * normalizePlanningContext()
 *        ↓
 * Planner
 *
 *
 * Этот модуль НЕ:
 *
 * - ищет Experience;
 * - хранит Skills;
 * - вызывает AI;
 * - изменяет память Jessica.
 *
 * Только нормализует данные.
 *
 * =========================================================
 */





/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


const MAX_ARRAY_ITEMS =
    20;







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


        plannerHints:
            [],


        metadata:
            {}

    };


}







/*
 * =========================================================
 * STRING
 * =========================================================
 */


function normalizeString(
    value
) {


    return typeof value === "string"

        ? value.trim()

        : String(
            value || ""
        )
        .trim();


}







/*
 * =========================================================
 * ARRAY
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
            normalizeString
        )

        .filter(
            Boolean
        )

        .slice(
            0,
            MAX_ARRAY_ITEMS
        );

}





/*
 * =========================================================
 * OBJECT
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




    /*
     * Поддержка старого формата:
     *
     * {
     *   experience:{
     *      ...
     *   }
     * }
     *
     */


    const source =


        experience.experience &&

        typeof experience.experience === "object"

            ? experience.experience

            : experience;






    return {



        /*
         * Идентификация Skill
         */


        skillId:

            normalizeString(

                source.skillId ||

                source.id

            ),



        name:

            normalizeString(
                source.name
            ),



        description:

            normalizeString(
                source.description
            ),



        version:

            Number(
                source.version || 1
            ),





        /*
         * Уровни доверия
         */


        skillConfidence:

            Number(
                source.skillConfidence ||
                source.confidence ||
                0
            ),



        matchConfidence:

            Number(
                source.matchConfidence ||
                experience.confidence ||
                0
            ),






        /*
         * Метаданные Skill
         */


        keywords:

            normalizeArray(
                source.keywords
            ),



        tags:

            normalizeArray(
                source.tags
            ),







        /*
         * Стратегия выполнения
         */


        strategy:

            normalizeArray(
                source.strategy
            ),



        sourcePriority:

            normalizeArray(
                source.sourcePriority
            ),



        validationRules:

            normalizeArray(
                source.validationRules
            ),



        failurePatterns:

            normalizeArray(
                source.failurePatterns
            ),



        successfulPatterns:

            normalizeArray(
                source.successfulPatterns
            ),



        avoidPatterns:

            normalizeArray(
                source.avoidPatterns
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
