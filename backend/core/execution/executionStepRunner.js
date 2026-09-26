/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v4
 * =========================================================
 *
 * Выполняет один проход Execution Cycle.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Task Runner
 *        ↓
 * Answer Composer
 *        ↓
 * Validator
 *        ↓
 * Execution Result
 *
 *
 * НЕ:
 *
 * - retry;
 * - replan;
 * - terminal decision;
 * - Learning save.
 *
 * =========================================================
 */


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
    buildCompletedResult
} from "./executionResult.js";








/*
 * =========================================================
 * STEP STATE
 * =========================================================
 */


function registerStep(

    context,

    stage,

    status,

    data = {}

) {


    if (
        !context
    ) {

        return;

    }



    context.currentStep =
        stage;



    if (
        !Array.isArray(
            context.stepsHistory
        )
    ) {

        context.stepsHistory = [];

    }



    context.stepsHistory.push({

        stage,

        status,


        ...data,


        timestamp:

            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * FAILURE BUILDER
 * =========================================================
 */


function createFailure(

    stage,

    reason,

    failureType,

    extra = {}

) {


    return {


        stage,


        failureType:


            failureType ||


            `${stage}-failure`,



        reason:

            reason ||

            "Ошибка выполнения",



        ...extra

    };

}









/*
 * =========================================================
 * EXCEPTION
 * =========================================================
 */


function exceptionResult(

    context,

    stage,

    error

) {


    registerStep(

        context,

        stage,

        "FAILED",

        {

            error:

                error?.message || ""

        }

    );



    return {


        success:false,


        failure:

            createFailure(

                stage,

                error?.message,

                `${stage}-exception`

            )


    };

}









/*
 * =========================================================
 * EXPERIENCE TRACE
 * =========================================================
 */


function attachExperience(

    result,

    context

) {


    if (
        !result
    ) {

        return result;

    }



    result.executionMeta = {


        ...(result.executionMeta || {}),



        experience:

        {


            used:

                context
                    ?.experience
                    ?.found === true,



            skills:

                context
                    ?.experience
                    ?.skills || []

        }


    };



    return result;

}









/*
 * =========================================================
 * MAIN EXECUTION
 * =========================================================
 */


export async function executeExecutionStep(

    context

) {



    /*
     * =====================================================
     * TASK RUNNER
     * =====================================================
     */


    registerStep(

        context,

        "runner",

        "RUNNING"

    );




    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        return exceptionResult(

            context,

            "runner",

            error

        );


    }






    const runFailure =

        analyzeRunFailure(

            context.runResult

        );





    if (
        runFailure?.failed === true
    ) {


        registerStep(

            context,

            "runner",

            "FAILED",

            {

                failure:

                    runFailure

            }

        );



        return {


            success:false,


            failure:

                runFailure

        };


    }







    registerStep(

        context,

        "runner",

        "COMPLETED"

    );









    /*
     * =====================================================
     * ANSWER COMPOSER
     * =====================================================
     */


    registerStep(

        context,

        "composer",

        "RUNNING"

    );



    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );



    } catch(error) {


        return exceptionResult(

            context,

            "composer",

            error

        );


    }







    if (
        !context.answerResult ||
        context.answerResult.success !== true
    ) {



        registerStep(

            context,

            "composer",

            "FAILED"

        );



        return {


            success:false,


            failure:

                createFailure(

                    "composer",

                    context.answerResult?.text ||

                    "Ответ не создан",

                    "composer-failure"

                )

        };


    }








    registerStep(

        context,

        "composer",

        "COMPLETED"

    );









    /*
     * =====================================================
     * VALIDATOR
     * =====================================================
     */


    registerStep(

        context,

        "validator",

        "RUNNING"

    );




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


        registerStep(

            context,

            "validator",

            "FAILED",

            {

                error:
                    error.message

            }

        );



        return {


            success:false,


            failure:

                createFailure(

                    "validator",

                    error.message,

                    "validator-exception"

                )


        };


    }







    context.validationResult =
        validation;







    if (
        validation?.valid === true
    ) {



        registerStep(

            context,

            "validator",

            "COMPLETED"

        );



        return {


            success:true,


            result:

                attachExperience(

                    buildCompletedResult(

                        context,

                        context.answerResult,

                        true

                    ),

                    context

                )


        };


    }









    registerStep(

        context,

        "validator",

        "FAILED",

        {

            validation

        }

    );






    return {


        success:false,


        failure:

            createFailure(

                "validator",

                validation?.reason ||

                "Результат не прошёл проверку",

                "validation-failure",

                {

                    validation

                }

            )


    };


}
