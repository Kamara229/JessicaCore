/*
 * =========================================================
 * JESSICA EXECUTION
 * EXECUTION LOOP v3
 * =========================================================
 *
 * Основной цикл выполнения.
 *
 *
 * Flow:
 *
 * Attempt
 *    ↓
 * Execute Step
 *    ↓
 * Analyze Result
 *    ↓
 * Retry / Replan / Finish
 *
 *
 * Ответственность:
 *
 * - управлять Execution iterations;
 * - передавать Context;
 * - обрабатывать решение Failure Handler.
 *
 *
 * НЕ:
 *
 * - создаёт Context;
 * - создаёт Trace;
 * - строит Plan;
 * - создаёт Terminal Result;
 * - реализует Replan;
 * - меняет Context напрямую.
 *
 * =========================================================
 */


import {
    executeExecutionStep
} from "../executionStepRunner.js";


import {
    handleExecutionFailure,
    FAILURE_ACTION
} from "../executionFailureHandler.js";


import {
    executeReplan
} from "./replanExecutor.js";


import {
    registerExecutionFailure,
    registerExecutionAttempt,
    registerExecutionRetry,
    getExecutionCounters
} from "../executionContext.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "../retryPolicy.js";









/*
 * =========================================================
 * FAILURE NORMALIZE
 * =========================================================
 */


function normalizeFailure(

    failure

) {


    return {


        stage:

            failure?.stage ||

            "execution",



        failureType:

            failure?.failureType ||

            "execution-error",



        reason:

            failure?.reason ||

            "Неизвестная ошибка"



    };

}









/*
 * =========================================================
 * EXECUTION EXCEPTION
 * =========================================================
 */


function buildExecutionException(

    error

) {


    return {


        stage:

            "execution",



        failureType:

            "execution-error",



        reason:

            error?.message ||

            "Execution exception"



    };

}









/*
 * =========================================================
 * LIMIT FAILURE
 * =========================================================
 */


function buildExecutionLimitFailure()

{


    return {


        stage:

            "execution",



        failureType:

            "execution-limit",



        reason:

            "Execution limit reached"



    };


}









/*
 * =========================================================
 * EXECUTION LOOP
 * =========================================================
 */


export async function executeExecutionLoop(

    context

) {


    let lastFailure = null;









    while (

        true

    ) {





        const counters =

            getExecutionCounters(

                context

            );









        if (

            counters.attempt >=

            MAX_EXECUTION_ATTEMPTS

        ) {


            break;

        }









        /*
         * =================================================
         * ATTEMPT
         * =================================================
         */


        registerExecutionAttempt(

            context,

            {

                retryCount:

                    counters.retryCount,



                replanCount:

                    counters.replanCount



            }

        );









        let result;









        /*
         * =================================================
         * EXECUTION STEP
         * =================================================
         */


        try {


            result =

                await executeExecutionStep(

                    context

                );



        }

        catch(error){


            result = {


                success:false,


                failure:

                    buildExecutionException(

                        error

                    )


            };


        }









        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (

            result?.success === true

        ) {


            return {


                success:true,


                result:

                    result.result



            };


        }









        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =

            normalizeFailure(

                result?.failure

            );









        registerExecutionFailure(

            context,

            lastFailure

        );









        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        if (

            !decision

        ) {


            break;

        }









        switch (

            decision.action

        ) {







            /*
             * =============================================
             * RETRY
             * =============================================
             */


            case FAILURE_ACTION.RETRY:


                registerExecutionRetry(

                    context

                );


                continue;









            /*
             * =============================================
             * REPLAN
             * =============================================
             */


            case FAILURE_ACTION.REPLAN:


                if (

                    await executeReplan(

                        context,

                        decision

                    )

                ) {


                    continue;


                }


                break;









            /*
             * =============================================
             * CLARIFICATION
             * =============================================
             */


            case FAILURE_ACTION.CLARIFICATION:


                return {


                    success:false,


                    failure:

                    {


                        ...lastFailure,


                        needsClarification:true



                    }



                };









            /*
             * =============================================
             * FINISH
             * =============================================
             */


            case FAILURE_ACTION.FINISH:


                break;


        }









        break;


    }









    return {


        success:false,


        failure:

            lastFailure ||

            buildExecutionLimitFailure()



    };


}
