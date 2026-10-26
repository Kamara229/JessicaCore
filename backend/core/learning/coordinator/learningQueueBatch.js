/*
 * =========================================================
 * JESSICA LEARNING QUEUE BATCH
 * =========================================================
 *
 * Последовательно сохраняет несколько Learning Events
 * одного Execution Trace.
 *
 * Каждый реально использованный Skill получает
 * собственный Queue Item.
 * =========================================================
 */


import {
    createLearningQueueItem
} from "../learningQueue.js";


import {
    saveLearningQueueItem
} from "../learningQueueStorage.js";


function isActionable(
    event
) {

    return Boolean(

        event

        &&

        event.action

        &&

        event.action !== "IGNORE"

    );

}


export async function persistLearningEvents(
    learningEvents,
    {saveQueueItem=saveLearningQueueItem}={}
) {

    const events =

        Array.isArray(
            learningEvents
        )

            ? learningEvents

            : [];


    const results = [];


    for(
        const event
        of events
    ){

        if(
            !isActionable(
                event
            )
        ){

            results.push({

                success:
                    true,

                queued:
                    false,

                ignored:
                    true,

                skillId:
                    event?.skillId || null,

                event

            });


            continue;

        }


        let queueItem;


        try {

            queueItem =

                createLearningQueueItem(
                    event
                );

        }catch(error){

            results.push({

                success:
                    false,

                queued:
                    false,

                skillId:
                    event?.skillId || null,

                error:

                    error?.message

                    ||

                    "Learning Queue Builder error",

                event

            });


            continue;

        }


        if(!queueItem){

            results.push({

                success:
                    false,

                queued:
                    false,

                skillId:
                    event?.skillId || null,

                error:
                    "Queue Item пустой",

                event

            });


            continue;

        }


        let storage;


        try {

            storage =

                await saveQueueItem(
                    queueItem
                );

        }catch(error){

            storage = {

                success:
                    false,

                error:

                    error?.message

                    ||

                    "Learning Queue Storage error"

            };

        }


        results.push({

            success:
                storage?.success === true,

            queued:
                storage?.success === true && storage?.queued !== false,

            existing:
                storage?.existing === true,

            skillId:
                event?.skillId || null,

            queueItem,

            storage,

            error:

                storage?.success === true

                    ? null

                    : (
                        storage?.error
                        ||
                        "Learning Queue не сохранён"
                    )

        });

    }


    return {

        success:

            results.every(
                item =>
                    item.success === true
            ),

        results,

        queuedCount:

            results.filter(
                item =>
                    item.queued === true
            )
            .length,

        ignoredCount:

            results.filter(
                item =>
                    item.ignored === true
            )
            .length,

        existingCount:

            results.filter(
                item => item.existing === true
            ).length,

        failedCount:

            results.filter(
                item =>
                    item.success !== true
            )
            .length,

        queueItems:

            results
                .filter(item => item.queued === true)
                .map(item => item.queueItem)
                .filter(Boolean)

    };

}
