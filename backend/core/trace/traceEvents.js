/*
 * =========================================================
 * JESSICA TRACE EVENTS v1
 * =========================================================
 *
 * История событий Execution Trace.
 *
 *
 * Ответственность:
 *
 * - Execution Events;
 * - Attempts;
 * - Replans.
 *
 *
 * НЕ:
 *
 * - создаёт Trace;
 * - формирует Result;
 * - завершает Trace;
 * - принимает Execution решения.
 *
 * =========================================================
 */


import {
    randomUUID
} from "node:crypto";



/*
 * =========================================================
 * EVENT
 * =========================================================
 */


export function addTraceEvent(

    trace,

    type,

    payload = {}

){


    if(!trace)
        return trace;



    if(
        !Array.isArray(
            trace.events
        )
    ){

        trace.events = [];

    }



    trace.events.push({

        id:

            randomUUID(),


        type,


        payload,


        timestamp:

            new Date()
            .toISOString()

    });



    return trace;

}



/*
 * =========================================================
 * ATTEMPT
 * =========================================================
 */


export function addTraceAttempt(

    trace,

    attempt,

    data = {}

){


    if(!trace)
        return trace;



    if(
        !trace.statistics ||
        typeof trace.statistics !== "object"
    ){

        trace.statistics = {

            attempts:0,

            retries:0,

            replans:0

        };

    }



    if(
        !Array.isArray(
            trace.attempts
        )
    ){

        trace.attempts = [];

    }



    trace.statistics.attempts =

        Number(
            attempt || 0
        );



    trace.attempts.push({

        attempt,


        ...data,


        timestamp:

            new Date()
            .toISOString()

    });



    return trace;

}



/*
 * =========================================================
 * REPLAN
 * =========================================================
 */


export function addTraceReplan(

    trace,

    data = {}

){


    if(!trace)
        return trace;



    if(
        !Array.isArray(
            trace.replans
        )
    ){

        trace.replans = [];

    }



    const item = {

        previousPlan:

            data.previousPlan ||
            null,


        newPlan:

            data.newPlan ||
            null,


        failure:

            data.failure ||
            null,


        timestamp:

            new Date()
            .toISOString()

    };



    trace.replans.push(
        item
    );



    addTraceEvent(

        trace,

        "REPLAN",

        item

    );



    return trace;

}
