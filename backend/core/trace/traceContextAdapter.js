/*
 * =========================================================
 * JESSICA TRACE CONTEXT ADAPTER v1
 * =========================================================
 *
 * Связь Execution Context и Execution Trace.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Trace Adapter
 *        ↓
 * Execution Trace
 *
 *
 * НЕ:
 *
 * - выполняет Execution;
 * - принимает решения;
 * - изменяет Context.
 *
 * =========================================================
 */


import {
    getExecutionCounters,
    getExecutionFailures,
    getExecutionHistory,
    getExecutionReplans
} from "../execution/context/contextReader.js";


import {
    addTraceEvent
} from "./executionTrace.js";









/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function safeArray(

    value

) {


    return Array.isArray(value)

        ?

        value

        :

        [];

}









/*
 * =========================================================
 * SYNC COUNTERS
 * =========================================================
 */


export function syncTraceCounters(

    trace,

    context

) {


    if (

        !trace ||

        !context

    ) {

        return trace;

    }









    const counters =

        getExecutionCounters(

            context

        );









    trace.statistics = {


        ...(trace.statistics || {}),



        attempts:

            counters.attempt,



        retries:

            counters.retryCount,



        replans:

            counters.replanCount


    };









    return trace;

}









/*
 * =========================================================
 * SYNC FAILURES
 * =========================================================
 */


export function syncTraceFailures(

    trace,

    context

) {


    if (

        !trace ||

        !context

    ) {

        return trace;

    }









    const failures =

        getExecutionFailures(

            context

        );









    trace.failures =

        safeArray(

            failures

        );









    return trace;

}









/*
 * =========================================================
 * SYNC REPLANS
 * =========================================================
 */


export function syncTraceReplans(

    trace,

    context

) {


    if (

        !trace ||

        !context

    ) {

        return trace;

    }









    trace.replans =

        safeArray(

            getExecutionReplans(

                context

            )

        );









    return trace;

}









/*
 * =========================================================
 * SYNC STEPS
 * =========================================================
 */


export function syncTraceSteps(

    trace,

    context

) {


    if (

        !trace ||

        !context

    ) {

        return trace;

    }









    trace.steps =

        safeArray(

            getExecutionHistory(

                context

            )

        );









    return trace;

}









/*
 * =========================================================
 * FULL SYNC
 * =========================================================
 */


export function syncTraceFromContext(

    trace,

    context

) {


    if (

        !trace ||

        !context

    ) {

        return trace;

    }









    syncTraceCounters(

        trace,

        context

    );









    syncTraceFailures(

        trace,

        context

    );









    syncTraceReplans(

        trace,

        context

    );









    syncTraceSteps(

        trace,

        context

    );









    addTraceEvent(

        trace,

        "CONTEXT_SYNCED",

        {

            attempt:

                trace.statistics?.attempts || 0,


            retries:

                trace.statistics?.retries || 0,


            replans:

                trace.statistics?.replans || 0

        }

    );









    return trace;

}
