/*
 * =========================================================
 * JESSICA EXECUTION TRACE FACADE v1
 * =========================================================
 *
 * Единая точка работы с Execution Trace.
 *
 *
 * Flow:
 *
 * Execution Event
 *        ↓
 * Trace Facade
 *        ↓
 * Execution Trace
 *
 *
 * Используется:
 *
 * - Jessica Core
 * - Execution Router
 * - Execution Coordinator
 *
 *
 * НЕ:
 *
 * - анализирует выполнение;
 * - принимает решения;
 * - изменяет Learning.
 *
 * =========================================================
 */


import {

    createExecutionTrace,

    updateTraceFromResult,

    updateTraceFromSummary,

    finishExecutionTrace

} from "./executionTrace.js";









/*
 * =========================================================
 * CREATE
 * =========================================================
 */


export function createJessicaExecutionTrace(

    task

)
{


    return createExecutionTrace(

        task

    );


}









/*
 * =========================================================
 * RESULT EVENT
 * =========================================================
 */


export function recordExecutionResult(

    trace,

    result

)
{


    if(
        !trace ||
        !result
    )
    {

        return trace;

    }




    return updateTraceFromResult(

        trace,

        result

    );


}









/*
 * =========================================================
 * SUMMARY EVENT
 * =========================================================
 */


export function recordExecutionSummary(

    trace,

    summary

)
{


    if(
        !trace ||
        !summary
    )
    {

        return trace;

    }



    return updateTraceFromSummary(

        trace,

        summary

    );


}









/*
 * =========================================================
 * FINISH
 * =========================================================
 */


export function completeExecutionTrace(

    trace

)
{


    if(
        !trace
    )
    {

        return trace;

    }



    return finishExecutionTrace(

        trace

    );


}
