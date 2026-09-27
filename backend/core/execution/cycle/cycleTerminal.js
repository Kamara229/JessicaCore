/*
 * =========================================================
 * JESSICA EXECUTION
 * CYCLE TERMINAL v4
 * =========================================================
 *
 * Финальное завершение Execution Cycle.
 *
 *
 * Flow:
 *
 * Context
 *    ↓
 * Build Result
 *    ↓
 * Attach Result
 *    ↓
 * Sync Trace
 *    ↓
 * Finish Context
 *    ↓
 * Finish Trace
 *    ↓
 * Return Result
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
    finishExecutionTrace,
    updateTraceFromResult
} from "../../trace/executionTrace.js";


import {
    syncTraceFromContext
} from "../../trace/traceContextAdapter.js";









/*
 * =========================================================
 * FINALIZE TRACE
 * =========================================================
 */


function finalizeTrace(

    context,

    result

){


    if(

        !context?.trace

    ){

        return;

    }









    /*
     * Result -> Trace
     */


    updateTraceFromResult(

        context.trace,

        result

    );









    /*
     * Context -> Trace
     */


    syncTraceFromContext(

        context.trace,

        context

    );









    finishExecutionTrace(

        context.trace

    );

}









/*
 * =========================================================
 * SAVE RESULT
 * =========================================================
 */


function saveTerminalResult(

    context,

    result

){


    if(!context)
        return;



    context.terminalResult = result;



    return result;

}









/*
 * =========================================================
 * FINISH FAILED
 * =========================================================
 */


export function finishFailedExecution(

    context,

    failure

){


    const result =

        buildTerminalResult(

            context,

            failure

        );









    saveTerminalResult(

        context,

        result

    );









    finishExecutionContext(

        context,

        "FAILED"

    );









    finalizeTrace(

        context,

        result

    );









    return result;

}









/*
 * =========================================================
 * FINISH SUCCESS
 * =========================================================
 */


export function finishSuccessfulExecution(

    context,

    result

){


    const finalResult =

        saveTerminalResult(

            context,

            result

        );









    finishExecutionContext(

        context,

        "COMPLETED"

    );









    finalizeTrace(

        context,

        finalResult

    );









    return finalResult;

}
