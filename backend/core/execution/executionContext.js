/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT v7
 * =========================================================
 *
 * Runtime context одного Execution Run.
 *
 *
 * Ответственность:
 *
 * - хранить состояние выполнения;
 * - хранить текущий Plan;
 * - хранить результаты этапов;
 * - хранить counters;
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


        executionId:

            randomUUID(),



        createdAt:

            now,



        startedAt:

            now,



        finishedAt:

            null,









        /*
         * STATE
         */

        state:

            "RUNNING",



        status:

            "ACTIVE",









        /*
         * INPUT
         */

        task:

            normalizeTask(
                task
            ),



        plan:

            plan || null,



        initialPlan:

            plan || null,



        planningContext:

            planningContext || {},







        /*
         * EXPERIENCE
         */

        experience:

            normalizeExperience(
                experience
            ),









        /*
         * EXECUTION DATA
         */

        currentStep:

            null,



        stepsHistory:

            [],



        completedSteps:

            [],



        failedSteps:

            [],



        executionHistory:

            [],









        /*
         * RESULTS
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
         * RETRY / REPLAN
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



        replans:

            [],







        /*
         * FAILURE
         */

        lastFailure:

            null,



        errors:

            [],









        /*
         * TRACE
         */

        trace:

            null,









        /*
         * LEARNING SIGNALS
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

        return context;

    }



    context.state =
        state;



    return context;

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





    context.currentStep =

        step.stage ||
        null;





    context.stepsHistory.push(
        record
    );





    context.executionHistory.push(
        record
    );








    if (
        step.status === "COMPLETED"
    ) {


        context.completedSteps.push(
            record
        );


    }








    if (
        step.status === "FAILED"
    ) {


        context.failedSteps.push(
            record
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





    const record = {


        ...data,


        timestamp:

            new Date()
                .toISOString()


    };





    context.replanHistory.push(
        record
    );



    context.replans.push(
        record
    );


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

            ?

            "COMPLETED"

            :

        state === "FAILED"

            ?

            "FAILED"

            :

            "FINISHED";





    context.finishedAt =

        new Date()
            .toISOString();


}
