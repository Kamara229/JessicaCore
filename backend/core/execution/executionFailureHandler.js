import {
    shouldRetryExecution
} from "./retryPolicy.js";


import {
    buildRunFailureFeedback,
    analyzeRunFailure
} from "./runFailurePolicy.js";


import {
    createAlternativePlan,
    applyAlternativePlan
} from "./replanCoordinator.js";


import {
    buildFailureResult,
    buildClarificationResult
} from "./executionResult.js";



/*
 * =========================================================
 * JESSICA EXECUTION FAILURE HANDLER
 * =========================================================
 *
 * Обрабатывает ошибки выполнения.
 *
 *
 * Flow:
 *
 * Failure
 *    ↓
 * Analyze
 *    ↓
 * Clarification?
 *    ↓
 * Retry?
 *    ↓
 * Replan
 *    ↓
 * Result
 *
 *
 * НЕ содержит:
 *
 * - TaskRunner
 * - Composer
 * - Validator
 * - Execution loop
 *
 * =========================================================
 */





/*
 * =========================================================
 * TRY REPLAN
 * =========================================================
 */


async function tryReplan(
    context,
    failure
) {


    const feedback =
        buildRunFailureFeedback(
            failure
        );



    const alternative =
        await createAlternativePlan(

            context,

            feedback

        );



    if (
        !alternative?.success
    ) {


        return {

            success:false,

            result:
                buildFailureResult(

                    context,

                    {

                        stage:
                            "replanner",

                        reason:
                            alternative?.reason ||
                            "Не удалось создать новый план",

                        failureType:
                            "replanner-failure"

                    }

                )

        };

    }




    const applied =
        applyAlternativePlan(

            context,

            alternative

        );



    if (
        applied !== true
    ) {


        return {

            success:false,

            result:
                buildFailureResult(

                    context,

                    {

                        stage:
                            "replanner",

                        reason:
                            "Не удалось применить новый план",

                        failureType:
                            "replan-apply-failure"

                    }

                )

        };

    }



    return {

        success:true

    };


}







/*
 * =========================================================
 * HANDLE FAILURE
 * =========================================================
 */


export async function handleExecutionFailure(

    context,

    failure

) {



    const analyzedFailure =
        analyzeRunFailure(
            failure
        )
        ||
        failure;





    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (
        analyzedFailure?.needsClarification === true
    ) {


        return {

            finished:true,

            result:
                buildClarificationResult(

                    context,

                    {

                        stage:
                            analyzedFailure.stage ||
                            "execution",

                        reason:
                            analyzedFailure.reason ||
                            "Требуется уточнение"

                    }

                )

        };

    }





    /*
     * =====================================================
     * RETRY CHECK
     * =====================================================
     */


    if (
        analyzedFailure?.shouldRetry !== true
    ) {


        return {

            finished:true,

            result:
                buildFailureResult(

                    context,

                    {

                        stage:
                            analyzedFailure.stage ||
                            "execution",

                        reason:
                            analyzedFailure.reason ||
                            "Не удалось выполнить задачу",

                        failureType:
                            analyzedFailure.failureType ||
                            "execution-failure"

                    }

                )

        };

    }





    /*
     * =====================================================
     * REPLAN
     * =====================================================
     */


    const replan =
        await tryReplan(

            context,

            analyzedFailure

        );



    if (
        !replan.success
    ) {


        return {

            finished:true,

            result:
                replan.result

        };

    }





    /*
     * =====================================================
     * CONTINUE LOOP
     * =====================================================
     */


    return {

        finished:false

    };


}
