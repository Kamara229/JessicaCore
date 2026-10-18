/*
 * =========================================================
 * JESSICA LEARNING COORDINATOR FACADE v2
 * =========================================================
 *
 * Execution Trace
 *      ↓
 * Learning Trigger
 *      ↓
 * one or more Learning Events
 *      ↓
 * Queue Batch Persistence
 *
 * Один complex Trace может усиливать
 * несколько реально применённых Skills.
 * =========================================================
 */


import {
    runLearningTrigger
} from "./learningTrigger.js";


import {
    persistLearningEvents
} from "./coordinator/learningQueueBatch.js";


function isValidTrace(
    trace
) {

    return Boolean(

        trace

        &&
        typeof trace === "object"

        &&
        !Array.isArray(trace)

    );

}


function buildFailureResult(
    reason
) {

    return {

        success:
            false,

        type:
            "LEARNING_EVENT",

        queued:
            false,

        queuedCount:
            0,

        reason

    };

}


export async function processLearning(
    trace
) {

    if(
        !isValidTrace(
            trace
        )
    ){

        return buildFailureResult(
            "Invalid execution trace"
        );

    }


    let triggerResult;


    try {

        triggerResult =

            runLearningTrigger(
                trace
            );

    }catch(error){

        console.error(

            "Jessica Learning Trigger error:",

            error

        );


        return buildFailureResult(
            "Ошибка запуска Learning Trigger"
        );

    }


    if(
        !triggerResult?.triggered
    ){

        return {

            success:
                true,

            type:
                "LEARNING_EVENT",

            queued:
                false,

            queuedCount:
                0,

            reason:

                triggerResult?.reason

                ||

                "Learning не запущен",

            trigger:
                triggerResult

        };

    }


    const learningEvents =

        Array.isArray(
            triggerResult.learningEvents
        )
        &&
        triggerResult.learningEvents.length > 0

            ? triggerResult.learningEvents

            : (
                triggerResult.learningEvent

                    ? [
                        triggerResult.learningEvent
                    ]

                    : []
            );


    if(
        learningEvents.length === 0
    ){

        return buildFailureResult(
            "Learning Event не создан"
        );

    }


    const persistence =

        await persistLearningEvents(
            learningEvents
        );


    const firstQueueItem =

        persistence.queueItems[0]

        ||

        null;


    console.log(

        "Jessica Learning Batch:",

        {

            events:
                learningEvents.length,

            queued:
                persistence.queuedCount,

            ignored:
                persistence.ignoredCount,

            failed:
                persistence.failedCount,

            skills:

                learningEvents
                    .map(
                        event =>
                            event?.skillId
                    )
                    .filter(Boolean)

        }

    );


    return {

        success:
            persistence.success,

        type:
            "LEARNING_EVENT",

        queued:
            persistence.queuedCount > 0,

        queuedCount:
            persistence.queuedCount,

        ignoredCount:
            persistence.ignoredCount,

        failedCount:
            persistence.failedCount,

        queueItem:
            firstQueueItem,

        queueItems:
            persistence.queueItems,

        batch:
            persistence.results,

        reason:

            persistence.success

                ? null

                : "Не все Learning Events сохранены",

        trigger:
            triggerResult

    };

}
