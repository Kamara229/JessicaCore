/*
 * =========================================================
 * JESSICA CONTEXT FACTORY v1
 * =========================================================
 *
 * Создание нового Execution Context.
 *
 *
 * Отвечает:
 *
 * - нормализация входных данных;
 * - создание runtime context.
 *
 *
 * НЕ:
 *
 * - меняет состояние;
 * - регистрирует шаги;
 * - хранит ошибки;
 * - управляет Execution Flow.
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

                ?

                experience.skills

                :

                [],



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
