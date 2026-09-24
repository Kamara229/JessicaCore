import {
    runPlan
} from "../taskRunner.js";


import {
    composeAnswer
} from "../answerComposer.js";


import {
    validateResult
} from "../validator.js";


import {
    analyzeRunFailure
} from "./runFailurePolicy.js";


import {
    buildCompletedResult,
    buildFailureResult
} from "./executionResult.js";



/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER
 * =========================================================
 *
 * Выполняет один цикл:
 *
 * Plan
 *  ↓
 * Runner
 *  ↓
 * Composer
 *  ↓
 * Validator
 *
 *
 * НЕ отвечает за:
 *
 * - retry
 * - replan
 * - лимиты попыток
 * - terminal outcome
 *
 * =========================================================
 */



export async function executeExecutionStep(
    context
) {


    /*
     * =====================================================
     * RUN PLAN
     * =====================================================
     */


    try {


        context.runResult =
            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        console.error(
            "Jessica TaskRunner error:",
            error
        );


        return {

            success:false,

            failure:{

                stage:
                    "runner",

                reason:
                    error?.message ||
                    "Ошибка выполнения плана",

                failureType:
                    "runner-exception"

            }

        };

    }





    /*
     * =====================================================
     * ANALYZE RUN
     * =====================================================
     */


    const runFailure =
        analyzeRunFailure(
            context.runResult
        );



    if (
        runFailure?.failed === true
    ) {


        return {

            success:false,

            failure:
                runFailure

        };

    }





    /*
     * =====================================================
     * COMPOSE ANSWER
     * =====================================================
     */


    try {


        context.answerResult =
            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );


    } catch(error) {


        return {

            success:false,

            failure:{

                stage:
                    "composer",

                reason:
                    error?.message ||
                    "Ошибка создания ответа",

                failureType:
                    "composer-exception"

            }

        };

    }





    if (
        !context.answerResult?.success
    ) {


        return {

            success:false,

            failure:{

                stage:
                    "composer",

                reason:
                    context.answerResult?.text ||
                    "Ответ не создан",

                failureType:
                    "composer-failure"

            }

        };

    }





    /*
     * =====================================================
     * VALIDATE
     * =====================================================
     */


    let validation;


    try {


        validation =
            await validateResult(

                context.task,

                context.plan,

                context.runResult,

                context.answerResult

            );


    } catch(error) {


        return {

            success:true,


            result:
                buildCompletedResult(

                    context,

                    context.answerResult,

                    false

                )

        };

    }





    /*
     * =====================================================
     * VALIDATION RESULT
     * =====================================================
     */


    if (
        validation?.valid === true
    ) {


        return {

            success:true,


            result:
                buildCompletedResult(

                    context,

                    context.answerResult,

                    true

                )

        };

    }





    return {

        success:false,

        failure:{

            stage:
                "validator",

            reason:
                validation?.reason ||
                "Ответ не прошёл проверку",

            validation

        }

    };


}
