/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT
 * =========================================================
 *
 * Контекст одной попытки выполнения.
 *
 *
 * Используется:
 *
 * - executionCycle
 * - executionStepRunner
 * - failureHandler
 * - ExecutionTrace
 *
 *
 * Хранит:
 *
 * - задачу;
 * - текущий Plan;
 * - PlanningContext;
 * - применённый Experience;
 * - результаты выполнения.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - делает retry;
 * - делает replan;
 * - валидирует результат.
 *
 * =========================================================
 */





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

        return {


            used:
                false,


            skills:
                [],


            context:
                null

        };

    }




    return {


        used:

            experience.used === true,



        skills:

            Array.isArray(
                experience.skills
            )

                ? experience.skills

                : [],



        context:

            experience.context || null


    };

}









/*
 * =========================================================
 * CREATE EXECUTION CONTEXT
 * =========================================================
 */


export function createExecutionContext({

    task,

    plan,

    planningContext = {},

    experience = null

} = {}) {


    return {



        /*
         * исходная задача
         */


        task,





        /*
         * текущий план
         */


        plan,





        /*
         * Контекст Planner
         *
         * Experience + rules
         */


        planningContext,





        /*
         * Experience,
         * реально использованный
         * при построении плана
         */


        experience:

            normalizeExperience(
                experience
            ),






        /*
         * результаты выполнения
         */


        runResult:

            null,



        answerResult:

            null,



        validationResult:

            null,






        /*
         * номер попытки
         */


        attempt:

            0,






        /*
         * история попыток
         */


        attempts:

            []



    };

}
