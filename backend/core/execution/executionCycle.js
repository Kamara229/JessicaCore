/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v13
 * =========================================================
 *
 * Central Execution Coordinator.
 *
 *
 * Ответственность:
 *
 * - создать Execution Context;
 * - создать Execution Trace;
 * - запустить Execution Loop;
 * - передать результат Terminal.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Context Factory
 *   ↓
 * Trace
 *   ↓
 * Execution Loop
 *   ↓
 * Terminal
 *   ↓
 * Result
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - анализирует ошибки;
 * - создаёт Answer;
 * - валидирует результат.
 *
 * =========================================================
 */


import {
    createExecutionContext
} from "./executionContext.js";


import {
    createExecutionTrace,
    addTraceEvent
} from "../trace/executionTrace.js";


import {
    executeExecutionLoop
} from "./cycle/executionLoop.js";


import {
    finishFailedExecution,
    finishSuccessfulExecution
} from "./cycle/cycleTerminal.js";









/*
 * =========================================================
 * SAFE FAILURE
 * =========================================================
 */


function buildExecutionException(

    error

) {


    return {


        stage:

            "execution",



        failureType:

            "execution-exception",



        reason:

            error?.message ||

            "Execution exception"


    };

}









/*
 * =========================================================
 * EXECUTE PLAN CYCLE
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {},

    experience = null

) {


    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    const context =

        createExecutionContext({

            task,

            plan:

                initialPlan,


            planningContext,


            experience

        });









    /*
     * =====================================================
     * TRACE
     * =====================================================
     */


    context.trace =

        createExecutionTrace(

            task

        );









    addTraceEvent(

        context.trace,

        "EXECUTION_STARTED",

        {

            executionId:

                context.executionId

        }

    );









    /*
     * =====================================================
     * LOOP
     * =====================================================
     */


    let result;



    try {


        result =

            await executeExecutionLoop(

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
     * =====================================================
     * TERMINAL
     * =====================================================
     */


    if(

        result?.success === true

    ){


        return finishSuccessfulExecution(

            context,

            result.result

        );


    }









    return finishFailedExecution(

        context,

        result?.failure ||

        {

            stage:

                "execution",



            failureType:

                "unknown",



            reason:

                "Unknown execution failure"

        }

    );


}
