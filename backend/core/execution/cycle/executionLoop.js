/*
 * =========================================================
 * JESSICA EXECUTION
 * EXECUTION LOOP v2
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
 * НЕ:
 *
 * - создаёт Context;
 * - создаёт Trace;
 * - строит Plan;
 * - создаёт Terminal Result;
 * - реализует Replan.
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
    registerExecutionAttempt
} from "../executionContext.js";


import {
    updateTraceFromResult,
    addTraceAttempt,
    addTraceEvent
} from "../../trace/executionTrace.js";


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
 * EXECUTION LOOP
 * =========================================================
 */


export async function executeExecutionLoop(

    context

) {


    let lastFailure = null;









    while(true){





        if(

            context.attempt >=

            MAX_EXECUTION_ATTEMPTS

        ){

            break;

        }









        registerExecutionAttempt(

            context,

            {

                retryCount:

                    context.retryCount,


                replanCount:

                    context.replanCount

            }

        );









        addTraceAttempt(

            context.trace,

            context.attempt,

            {

                retryCount:

                    context.retryCount,


                replanCount:

                    context.replanCount

            }

        );









        let result;









        try {


            result =

                await executeExecutionStep(

                    context

                );


        }

        catch(error){


            result = {


                success:false,


                failure:{

                    stage:

                        "execution",



                    failureType:

                        "execution-error",



                    reason:

                        error?.message ||

                        "Execution exception"

                }

            };


        }









        updateTraceFromResult(

            context.trace,

            result

        );









        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if(

            result?.success === true

        ){


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









        switch(

            decision.action

        ){





            case FAILURE_ACTION.RETRY:


                context.retryCount++;


                addTraceEvent(

                    context.trace,

                    "RETRY"

                );


                continue;









            case FAILURE_ACTION.REPLAN:


                if(

                    await executeReplan(

                        context,

                        decision

                    )

                ){

                    continue;

                }


                break;









            case FAILURE_ACTION.CLARIFICATION:


                return {


                    success:false,


                    failure:{

                        ...lastFailure,


                        needsClarification:true

                    }

                };









            case FAILURE_ACTION.FINISH:


                break;


        }






        break;


    }









    return {


        success:false,


        failure:

            lastFailure ||

            {

                stage:

                    "execution",



                failureType:

                    "execution-limit",



                reason:

                    "Execution limit reached"

            }


    };


}
