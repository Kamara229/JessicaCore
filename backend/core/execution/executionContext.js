/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT v3
 * =========================================================
 *
 * Контекст одного Execution Cycle.
 *
 *
 * Flow:
 *
 * Planner
 *    ↓
 * Execution Context
 *    ↓
 * Execution Cycle
 *    ↓
 * Execution Trace
 *    ↓
 * Learning
 *
 *
 * Используется:
 *
 * - executionCycle
 * - executionStepRunner
 * - failureHandler
 * - ExecutionTrace
 * - Learning Analyzer
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



import {
    randomUUID
} from "node:crypto";






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


            found:
                false,


            source:
                null,


            confidence:
                0,


            skills:
                [],


            context:
                null


        };

    }




    return {


        found:

            experience.found === true,



        source:

            experience.source ||
            null,



        confidence:

            Number(
                experience.confidence || 0
            ),



        skills:

            Array.isArray(
                experience.skills
            )

                ? experience.skills

                : [],



        context:

            experience.context ||
            null


    };


}









/*
 * =========================================================
 * NORMALIZE PLANNER TRACE
 * =========================================================
 */


function normalizePlannerTrace(
    trace
) {


    return Array.isArray(
        trace
    )
        ? trace
        : [];

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

    experience = null,

    plannerTrace = []

} = {}) {



    const executionId =
        randomUUID();




    return {



        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        executionId,



        createdAt:

            new Date()
                .toISOString(),



        startedAt:

            new Date()
                .toISOString(),



        finishedAt:

            null,





        /*
         * =================================================
         * STATE
         * =================================================
         */


        state:

            "RUNNING",





        /*
         * =================================================
         * TASK
         * =================================================
         */


        task:

            String(
                task || ""
            )
            .trim(),





        /*
         * =================================================
         * PLAN
         * =================================================
         */


        plan:

            plan || null,





        plannerTrace:

            normalizePlannerTrace(
                plannerTrace
            ),






        /*
         * =================================================
         * PLANNING CONTEXT
         * =================================================
         */


        planningContext:

            planningContext || {},






        /*
         * =================================================
         * EXPERIENCE
         * =================================================
         *
         * Опыт, доступный Planner.
         *
         */


        experience:

            normalizeExperience(
                experience
            ),






        /*
         * =================================================
         * EXECUTION DATA
         * =================================================
         */


        currentStep:

            null,



        completedSteps:

            [],



        failedSteps:

            [],



        stepsHistory:

            [],






        /*
         * =================================================
         * RESULTS
         * =================================================
         */


        runResult:

            null,



        answerResult:

            null,



        validationResult:

            null,







        /*
         * =================================================
         * ATTEMPTS
         * =================================================
         */


        attempt:

            0,



        attempts:

            [],







        /*
         * =================================================
         * ERRORS
         * =================================================
         */


        errors:

            [],







        /*
         * =================================================
         * LEARNING DATA
         * =================================================
         *
         * Передаётся Experience Analyzer.
         *
         */


        learningContext:

            {


                reusable:
                    false,


                candidateSkill:
                    null,


                signals:
                    []

            }





    };

}
