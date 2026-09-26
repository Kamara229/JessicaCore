/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT v5
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
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - валидирует Answer;
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
 * CREATE CONTEXT
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
         * EXECUTION TRACKING
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



        terminalResult:

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



        replanHistory:

            [],



        attempts:

            [],







        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure:

            null,



        errors:

            [],









        /*
         * =================================================
         * TRACE
         * =================================================
         */


        trace:

            null,









        /*
         * =================================================
         * LEARNING
         * =================================================
         */


        learningContext:

        {


            signals:

                [],



            successful:

                false,



            reusable:

                false,



            candidateSkill:

                null


        }



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

    step

) {


    if (
        !context ||
        !step
    ) {

        return;

    }





    const record = {


        ...step,



        timestamp:

            new Date()
                .toISOString()


    };






    context.stepsHistory.push(
        record
    );






    if (
        step.status === "COMPLETED"
    ) {


        context.completedSteps.push(
            step
        );


    }






    if (
        step.status === "FAILED"
    ) {


        context.failedSteps.push(
            step
        );


    }


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
 * REGISTER REPLAN
 * =========================================================
 */


export function registerReplan(

    context,

    data

) {


    if (
        !context ||
        !data
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




    context.status =


        state === "COMPLETED"

            ? "COMPLETED"

            :

        state === "FAILED"

            ? "FAILED"

            :

            "FINISHED";






    context.finishedAt =

        new Date()
            .toISOString();


}
