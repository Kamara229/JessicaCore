/*
 * =========================================================
 * JESSICA PLANNING CONTEXT
 * =========================================================
 *
 * Единый формат контекста Planner.
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
 * НЕ:
 *
 * - ищет Experience;
 * - изменяет Skills;
 * - вызывает AI.
 *
 * Только приводит данные к единому виду.
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


        plannerHints:
            [],


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

                typeof item === "string"

                    ? item.trim()

                    : item

        )

        .filter(Boolean);

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
 * NORMALIZE SKILL
 * =========================================================
 */


function normalizeSkill(
    skill
) {


    if (
        !skill ||
        typeof skill !== "object"
    ) {

        return null;

    }


    return {


        id:
            skill.id ||
            null,


        name:
            String(
                skill.name || ""
            )
            .trim(),



        workflow:

            normalizeArray(
                skill.workflow
            ),



        constraints:

            normalizeArray(
                skill.constraints
            ),



        examples:

            normalizeArray(
                skill.examples
            )

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






    const nestedExperience =

        experience.experience &&

        typeof experience.experience === "object"

            ? experience.experience

            : {};







    return {


        /*
         * Новая модель памяти
         */


        found:

            experience.found === true,



        source:

            experience.source ||
            "unknown",



        confidence:

            Number(
                experience.confidence || 0
            ),



        skills:

            Array.isArray(
                nestedExperience.skills
            )

                ? nestedExperience.skills

                    .map(
                        normalizeSkill
                    )

                    .filter(Boolean)

                : [],





        /*
         * Старые поля Experience
         * оставляем для совместимости
         */


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
