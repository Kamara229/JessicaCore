/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT v6
 * =========================================================
 *
 * Runtime context одного Execution Run.
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
 * Ответственность:
 *
 * - хранить состояние выполнения;
 * - хранить текущий Plan;
 * - хранить результаты этапов;
 * - хранить retry/replan counters;
 * - передавать данные между Execution модулями.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - принимает решения;
 * - делает Retry;
 * - делает Replan;
 * - валидирует Answer;
 * - сохраняет Learning.
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


            used:
                false,


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


        used:

            experience.used === true,



        found:

            experience.found === true,



        source:

            experience.source || null,



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

            experience.context || null


    };

}









/*
 * =========================================================
 * NORMALIZE TASK
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
         * EXPERIENCE SNAPSHOT
         * =================================================
         */


        experience:

            normalizeExperience(
                experience
            ),







        /*
         * =================================================
         * EXECUTION RESULTS
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
         * RETRY / REPLAN STATE
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



        replanHistory:

            [],









        /*
         * =================================================
         * FAILURE STATE
         * =================================================
         */


        lastFailure:

            null,



        errors:

            [],







        /*
         * =================================================
         * TRACE REFERENCE
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
        !context
    ) {

        return context;

    }



    context.state =
        state;



    return context;

}









/*
 * =========================================================
 * REGISTER FAILURE
 * =========================================================
 */


export function registerExecutionFailure(

    context,

    failure

) {


    if (
        !context ||
        !failure
    ) {

        return;

    }





    context.lastFailure =
        failure;





    context.errors.push({

        ...failure,


        timestamp:

            new Date()
                .toISOString()


    });


}









/*
 * =========================================================
 * REGISTER ATTEMPT
 * =========================================================
 */


export function registerExecutionAttempt(

    context,

    data = {}

) {


    if (
        !context
    ) {

        return;

    }





    context.attempt++;





    context.attempts.push({

        attempt:

            context.attempt,



        ...data,



        timestamp:

            new Date()
                .toISOString()


    });


}









/*
 * =========================================================
 * REGISTER REPLAN
 * =========================================================
 */


export function registerExecutionReplan(

    context,

    data = {}

) {


    if (
        !context
    ) {

        return;

    }





    context.replanCount++;





    context.replanHistory.push({

        ...data,



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



    context.finishedAt =

        new Date()
            .toISOString();


}
