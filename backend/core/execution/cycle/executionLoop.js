/*
 * =========================================================
 * JESSICA EXECUTION
 * EXECUTION LOOP v4
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
 * Failure Handler
 *    ↓
 *
 * Retry
 * Replan
 * Finish
 *
 *
 * НЕ:
 *
 * - создаёт Context;
 * - создаёт Trace;
 * - строит Plan;
 * - создаёт Terminal Result;
 * - принимает Failure Decision.
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
    addTraceEvent
} from "../../trace/executionTrace.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "../retryPolicy.js";









/*
 * =========================================================
 * EXCEPTION
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
 * LIMIT
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









    while(true){





        const counters =

            getExecutionCounters(

                context

            );









        if(

            counters.attempt >=

            MAX_EXECUTION_ATTEMPTS

        ){

            break;

        }









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
         * Failure передаётся
         * без изменения
         */


        lastFailure =

            result?.failure ||

            buildExecutionLimitFailure();









        registerExecutionFailure(

            context,

            lastFailure

        );









        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        if(

            !decision

        ){

            break;

        }









        switch(

            decision.action

        ){



            case FAILURE_ACTION.RETRY:


                registerExecutionRetry(

                    context

                );


                addTraceEvent(

                    context.trace,

                    "RETRY",

                    {

                        attempt:

                            context.attempt

                    }

                );


                continue;









            case FAILURE_ACTION.REPLAN:


                if(

                    await executeReplan(

                        context,

                        decision

                    )

                ){


                    addTraceEvent(

                        context.trace,

                        "REPLAN_CONTINUE"

                    );


                    continue;


                }


                break;









            case FAILURE_ACTION.CLARIFICATION:


                return {


                    success:false,


                    failure:

                        lastFailure


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

            buildExecutionLimitFailure()



    };


}
