/*
 * =========================================================
 * JESSICA CONTEXT FACTORY v3
 * =========================================================
 *
 * Создание нового Execution Context.
 *
 *
 * Отвечает:
 *
 * - нормализация входных данных;
 * - создание полной runtime-схемы Context;
 * - инициализация histories и counters.
 *
 *
 * НЕ:
 *
 * - меняет состояние после создания;
 * - регистрирует шаги;
 * - регистрирует ошибки;
 * - делает Retry;
 * - делает Replan;
 * - управляет Execution Flow.
 *
 * =========================================================
 */


import {
    randomUUID
} from "node:crypto";









/*
 * =========================================================
 * SAFE NUMBER
 * =========================================================
 */


function safeNumber(

    value

) {


    const number =

        Number(

            value

        );



    return Number.isFinite(

        number

    )

        ?

        number

        :

        0;

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


        return {


            used:false,


            found:false,


            source:null,


            confidence:0,


            skills:[],


            context:null


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

            safeNumber(

                experience.confidence

            ),



        skills:

            Array.isArray(

                experience.skills

            )

                ?

                [
                    ...experience.skills
                ]

                :

                [],



        context:

            experience.context ||

            null


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
 * NORMALIZE PLANNING CONTEXT
 * =========================================================
 */


function normalizePlanningContext(

    planningContext

) {


    if (

        !planningContext ||

        typeof planningContext !== "object" ||

        Array.isArray(planningContext)

    ) {


        return {};

    }



    return {

        ...planningContext

    };

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

            plan ||

            null,



        initialPlan:

            plan ||

            null,



        planningContext:

            normalizePlanningContext(

                planningContext

            ),









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
         * EXECUTION STEPS
         * =================================================
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



        resultHistory:

            [],









        /*
         * =================================================
         * COUNTERS
         * =================================================
         */


        attempt:

            0,



        retryCount:

            0,



        replanCount:

            0,









        /*
         * =================================================
         * ATTEMPT HISTORY
         * =================================================
         */


        attempts:

            [],









        /*
         * =================================================
         * RETRY HISTORY
         * =================================================
         */


        retryHistory:

            [],









        /*
         * =================================================
         * REPLAN HISTORY
         * =================================================
         */


        replanHistory:

            [],



        /*
         * Compatibility / secondary history.
         *
         * Пока сохраняем, потому что существующий
         * Context API использует context.replans.
         *
         * После проверки contextReplan.js и
         * contextReader.js решим, нужен ли этот
         * второй массив вообще.
         */


        replans:

            [],









        /*
         * =================================================
         * FAILURES
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


            signals:[],



            successful:false,



            reusable:false,



            candidateSkill:null


        },









        /*
         * =================================================
         * METADATA
         * =================================================
         */


        metadata:

        {


            version:

                "context-v3"


        }


    };

}
