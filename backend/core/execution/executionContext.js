/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT v4
 * =========================================================
 *
 * Контекст одного Execution Run.
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
 * - executionTrace
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - делает retry;
 * - делает replan;
 * - валидирует результат;
 * - обучает Jessica.
 *
 * =========================================================
 */


import {
    randomUUID
} from "node:crypto";








/*
 * =========================================================
 * EXPERIENCE NORMALIZER
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
 * TASK NORMALIZER
 * =========================================================
 */


function normalizeTask(
    task
) {


    return String(
        task || ""
    )
    .trim();

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



    const now =
        new Date()
            .toISOString();



    return {


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        executionId:

            randomUUID(),



        createdAt:

            now,



        startedAt:

            now,



        finishedAt:

            null,






        /*
         * =================================================
         * STATE
         * =================================================
         */


        state:

            "RUNNING",



        status:

            "ACTIVE",








        /*
         * =================================================
         * INPUT
         * =================================================
         */


        task:

            normalizeTask(
                task
            ),



        plan:

            plan || null,



        planningContext:

            planningContext || {},







        /*
         * =================================================
         * EXPERIENCE
         * =================================================
         */


        experience:

            normalizeExperience(
                experience
            ),








        /*
         * =================================================
         * EXECUTION POSITION
         * =================================================
         */


        currentStep:

            null,



        completedSteps:

            [],



        failedSteps:

            [],



        executionHistory:

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
         * RETRY / REPLAN
         * =================================================
         */


        attempt:

            0,



        retryCount:

            0,



        replanCount:

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
         * TRACE HOLDER
         * =================================================
         */


        trace:

            null





    };

}









/*
 * =========================================================
 * UPDATE STATE
 * =========================================================
 */


export function updateExecutionState(

    context,

    state

) {


    if (
        !context ||
        typeof context !== "object"
    ) {

        return;

    }



    context.state =
        state;


}









/*
 * =========================================================
 * REGISTER STEP
 * =========================================================
 */


export function registerExecutionStep(

    context,

    stepResult

) {


    if (
        !context ||
        !stepResult
    ) {

        return;

    }



    context.executionHistory.push({

        ...stepResult,

        timestamp:

            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * REGISTER ERROR
 * =========================================================
 */


export function registerExecutionError(

    context,

    error

) {


    if (
        !context ||
        !error
    ) {

        return;

    }



    context.errors.push({

        ...error,

        timestamp:

            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * FINISH CONTEXT
 * =========================================================
 */


export function finishExecutionContext(
    context,
    state = "FINISHED"
) {


    if (
        !context
    ) {

        return;

    }



    context.state =
        state;



    context.status =
        "COMPLETED";



    context.finishedAt =

        new Date()
            .toISOString();


}
