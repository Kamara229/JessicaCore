/*
 * =========================================================
 * JESSICA EXECUTION
 * CYCLE TERMINAL v2
 * =========================================================
 *
 * Финальное завершение Execution Cycle.
 *
 *
 * Flow:
 *
 * Context
 *    ↓
 * Sync Trace
 *    ↓
 * Finish Context
 *    ↓
 * Finish Trace
 *    ↓
 * Terminal Result
 *
 *
 * НЕ:
 *
 * - выполняет Execution;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    finishExecutionContext
} from "../executionContext.js";


import {
    buildTerminalResult
} from "../executionTerminal.js";


import {
    finishExecutionTrace
} from "../../trace/executionTrace.js";


import {
    syncTraceFromContext
} from "../../trace/traceContextAdapter.js";


import {
    getExecutionId,
    getTraceId,
    getExecutionCounters
} from "../context/contextReader.js";









/*
 * =========================================================
 * META
 * =========================================================
 */


function attachExecutionMeta(

    result,

    context

) {


    const counters =

        getExecutionCounters(

            context

        );



    return {


        ...result,


        executionMeta:

        {


            ...(result?.executionMeta || {}),



            executionId:

                getExecutionId(

                    context

                ),



            traceId:

                getTraceId(

                    context

                ),



            attempt:

                counters.attempt,



            retryCount:

                counters.retryCount,



            replanCount:

                counters.replanCount


        }


    };

}









/*
 * =========================================================
 * PREPARE TRACE
 * =========================================================
 */


function prepareTrace(

    context

) {


    if (

        !context?.trace

    ) {

        return;

    }



    syncTraceFromContext(

        context.trace,

        context

    );

}









/*
 * =========================================================
 * FINISH FAILED
 * =========================================================
 */


export function finishFailedExecution(

    context,

    failure

) {


    prepareTrace(

        context

    );



    finishExecutionContext(

        context,

        "FAILED"

    );



    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        buildTerminalResult(

            context,

            failure

        ),

        context

    );


}









/*
 * =========================================================
 * FINISH SUCCESS
 * =========================================================
 */


export function finishSuccessfulExecution(

    context,

    result

) {


    prepareTrace(

        context

    );



    finishExecutionContext(

        context,

        "COMPLETED"

    );



    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        result,

        context

    );


}
