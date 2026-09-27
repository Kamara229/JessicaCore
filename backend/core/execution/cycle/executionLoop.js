/*
 * =========================================================
 * JESSICA EXECUTION
 * EXECUTION LOOP v1
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
 * - создаёт Terminal Result.
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
    createAlternativePlan,
    applyAlternativePlan
} from "../replanCoordinator.js";


import {
    registerExecutionFailure,
    registerExecutionAttempt
} from "../executionContext.js";


import {
    updateTraceFromResult,
    addTraceAttempt,
    addTraceEvent,
    addTraceReplan
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
 * REPLAN
 * =========================================================
 */


async function executeReplan(

    context,

    decision

) {


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:

                decision.failure

        }

    );




    const previousPlan =

        context.plan;




    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );





    if(
        !alternative?.success
    ){

        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            alternative

        );


        return false;

    }





    const applied =

        applyAlternativePlan(

            context,

            alternative

        );





    if(!applied){

        return false;

    }





    context.attempt = 0;

    context.retryCount = 0;




    addTraceReplan(

        context.trace,

        {

            previousPlan,

            newPlan:
                context.plan,

            failure:
                decision.failure

        }

    );




    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED"

    );



    return true;

}









/*
 * =========================================================
 * EXECUTE LOOP
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






        if(

            result?.success === true

        ){


            return {


                success:true,


                result:

                    result.result


            };


        }






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
