/*
 * =========================================================
 * JESSICA TRACE CONTEXT ADAPTER v3
 * =========================================================
 *
 * Связь Execution Context
 * и Execution Trace.
 *
 *
 * Flow:
 *
 * Execution Context
 *        ↓
 * Trace Context Adapter
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
    getExecutionReplans,
    getExperience
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

        ? value

        : [];

}


/*
 * =========================================================
 * SAFE NUMBER
 * =========================================================
 */


function safeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


/*
 * =========================================================
 * COUNTERS
 * =========================================================
 */


export function syncTraceCounters(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

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
 * FAILURES
 * =========================================================
 */


export function syncTraceFailures(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

        return trace;

    }


    trace.failures = [

        ...safeArray(

            getExecutionFailures(
                context
            )

        )

    ];


    return trace;

}


/*
 * =========================================================
 * REPLANS
 * =========================================================
 */


export function syncTraceReplans(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

        return trace;

    }


    trace.replans = [

        ...safeArray(

            getExecutionReplans(
                context
            )

        )

    ];


    return trace;

}


/*
 * =========================================================
 * STEPS
 * =========================================================
 */


export function syncTraceSteps(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

        return trace;

    }


    trace.steps = [

        ...safeArray(

            getExecutionHistory(
                context
            )

        )

    ];


    return trace;

}


/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


export function syncTraceExperience(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

        return trace;

    }


    const experience =

        getExperience(
            context
        )

        ||

        {};


    const skills =

        safeArray(
            experience.skills
        )

        .map(

            skill =>

                skill
                &&
                typeof skill === "object"

                    ? {
                        ...skill
                    }

                    : skill

        );


    trace.experienceUsage = {

        ...(trace.experienceUsage || {}),

        used:

            experience.used === true,

        found:

            experience.found === true,

        source:

            experience.source

            ||

            null,

        confidence:

            safeNumber(
                experience.confidence
            ),

        skills

    };


    return trace;

}


/*
 * =========================================================
 * RESULT
 * =========================================================
 */


export function syncTraceResult(

    trace,

    context

) {

    if(
        !trace
        ||
        !context
    ){

        return trace;

    }


    if(
        context.terminalResult
    ){

        trace.result =
            context.terminalResult;


        trace.terminal =

            context.terminalResult.terminal

            ||

            null;

    }


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

    if(
        !trace
        ||
        !context
    ){

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


    syncTraceExperience(
        trace,
        context
    );


    syncTraceResult(
        trace,
        context
    );


    addTraceEvent(

        trace,

        "CONTEXT_SYNCED",

        {

            attempt:

                trace.statistics?.attempts

                ||

                0,

            retries:

                trace.statistics?.retries

                ||

                0,

            replans:

                trace.statistics?.replans

                ||

                0,

            experienceUsed:

                trace.experienceUsage?.used === true,

            experienceFound:

                trace.experienceUsage?.found === true,

            experienceSkills:

                safeArray(
                    trace.experienceUsage?.skills
                )
                .length

        }

    );


    return trace;

}
