/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v11
 * =========================================================
 *
 * Central Execution Coordinator.
 *
 *
 * Ответственность:
 *
 * Create Context
 * Create Trace
 * Start Execution Loop
 * Finish Result
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Context
 *   ↓
 * Trace
 *   ↓
 * Execution Loop
 *   ↓
 * Terminal
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - анализирует ошибки;
 * - создаёт Answer;
 * - валидирует.
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
 * EXECUTE PLAN CYCLE
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

) {


    /*
     * =====================================================
     * CREATE CONTEXT
     * =====================================================
     */


    const context =

        createExecutionContext({

            task,

            plan:
                initialPlan,


            planningContext

        });









    /*
     * =====================================================
     * CREATE TRACE
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
     * START LOOP
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


            failure:{

                stage:
                    "execution",


                failureType:
                    "execution-exception",


                reason:

                    error?.message ||

                    "Execution exception"

            }


        };


    }









    /*
     * =====================================================
     * SUCCESS
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









    /*
     * =====================================================
     * FAILURE
     * =====================================================
     */


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
