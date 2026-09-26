/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER v3
 * =========================================================
 *
 * Выполняет один полный Execution Step.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Task Runner
 *   ↓
 * Answer Composer
 *   ↓
 * Validator
 *   ↓
 * Step Result
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
    buildCompletedResult,
    buildFailureResult
} from "./executionResult.js";








/*
 * =========================================================
 * REGISTER STEP
 * =========================================================
 */


function registerStep(

    context,

    {

        stage,

        status,

        error = null

    }

) {


    if (
        !context
    ) {

        return;

    }



    context.currentStep =
        stage;



    context.stepsHistory.push({

        stage,

        status,

        error,

        timestamp:
            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * REGISTER LEARNING SIGNAL
 * =========================================================
 */


function addLearningSignal(

    context,

    signal

) {


    if (
        !context?.learningContext
    ) {

        return;

    }



    if (
        !Array.isArray(
            context.learningContext.signals
        )
    ) {

        context.learningContext.signals = [];

    }



    context.learningContext.signals.push(
        signal
    );

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

        reason:

            reason ||

            "Ошибка выполнения",


        failureType:


            failureType ||


            `${stage}-failure`,



        ...extra

    };

}









/*
 * =========================================================
 * EXCEPTION HANDLER
 * =========================================================
 */


function handleException(

    context,

    stage,

    error

) {


    registerStep(

        context,

        {

            stage,

            status:
                "FAILED",

            error:
                error?.message

        }

    );



    addLearningSignal(

        context,

        {

            type:
                "execution_exception",


            stage,


            reason:
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
 * EXECUTE EXECUTION STEP
 * =========================================================
 */


export async function executeExecutionStep(

    context

) {



    /*
     * =====================================================
     * RUNNER
     * =====================================================
     */


    registerStep(

        context,

        {

            stage:
                "runner",

            status:
                "RUNNING"

        }

    );



    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        return handleException(

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

            {

                stage:
                    "runner",

                status:
                    "FAILED"

            }

        );



        addLearningSignal(

            context,

            {

                type:
                    "runner_failure",


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

        {

            stage:
                "runner",

            status:
                "COMPLETED"

        }

    );









    /*
     * =====================================================
     * COMPOSER
     * =====================================================
     */


    registerStep(

        context,

        {

            stage:
                "composer",

            status:
                "RUNNING"

        }

    );



    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );


    } catch(error) {


        return handleException(

            context,

            "composer",

            error

        );

    }








    if (
        !context.answerResult?.success
    ) {


        registerStep(

            context,

            {

                stage:
                    "composer",

                status:
                    "FAILED"

            }

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

        {

            stage:
                "composer",

            status:
                "COMPLETED"

        }

    );









    /*
     * =====================================================
     * VALIDATOR
     * =====================================================
     */


    registerStep(

        context,

        {

            stage:
                "validator",

            status:
                "RUNNING"

        }

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



        /*
         * Ошибка самого Validator.
         *
         * Ответ существует,
         * но подтверждение отсутствует.
         */


        registerStep(

            context,

            {

                stage:
                    "validator",

                status:
                    "SKIPPED"

            }

        );



        addLearningSignal(

            context,

            {

                type:
                    "validation_skipped"

            }

        );



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






    context.validationResult =
        validation;







    if (
        validation?.valid === true
    ) {


        registerStep(

            context,

            {

                stage:
                    "validator",

                status:
                    "COMPLETED"

            }

        );



        addLearningSignal(

            context,

            {

                type:
                    "successful_execution",


                validated:
                    true

            }

        );



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







    /*
     * =====================================================
     * VALIDATION FAILURE
     * =====================================================
     */


    registerStep(

        context,

        {

            stage:
                "validator",

            status:
                "FAILED"

        }

    );



    addLearningSignal(

        context,

        {

            type:
                "validation_failure",


            reason:
                validation?.reason || ""

        }

    );



    return {


        success:false,


        failure:

            createFailure(

                "validator",

                validation?.reason ||

                "Ответ не прошёл проверку",

                "validation-failure",

                {

                    validation

                }

            )


    };


}
