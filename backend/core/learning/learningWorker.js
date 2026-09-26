/*
 * =========================================================
 * JESSICA LEARNING WORKER
 * =========================================================
 *
 * Обработчик очереди обучения.
 *
 *
 * Flow:
 *
 * learning_queue
 *        ↓
 * Learning Worker
 *        ↓
 * Learning Proposal
 *        ↓
 * Learning Approval
 *
 *
 * НЕ:
 *
 * - создаёт Experience Skill;
 * - пишет напрямую в Experience Storage;
 * - принимает решение Approval.
 *
 * =========================================================
 */


import {
    getPendingLearningItems,
    updateLearningQueueItemStatus
} from "./learningQueueStorage.js";


import {
    createLearningProposal
} from "./learningProposal.js";





/*
 * =========================================================
 * PROCESS SINGLE ITEM
 * =========================================================
 */


async function processQueueItem(
    item
) {


    if (
        !item
    ) {

        return {

            success:
                false,

            reason:
                "Empty queue item"

        };

    }



    try {


        const proposal =
            createLearningProposal(
                item
            );



        if (
            !proposal
        ) {


            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );


            return {

                success:
                    false,

                reason:
                    "Proposal not created"

            };

        }



        await updateLearningQueueItemStatus(

            item.id,

            "PROPOSED"

        );



        return {

            success:
                true,

            proposal


        };



    } catch(error) {


        console.error(

            "Jessica Learning Worker error:",

            error

        );



        await updateLearningQueueItemStatus(

            item.id,

            "FAILED"

        );



        return {

            success:
                false,

            reason:
                error.message

        };

    }


}





/*
 * =========================================================
 * RUN WORKER
 * =========================================================
 */


export async function runLearningWorker()
{


    const queueResult =
        await getPendingLearningItems();



    if (
        !queueResult?.success
    ) {


        return {

            success:
                false,

            processed:
                0,

            error:
                queueResult?.error

        };

    }



    const items =
        queueResult.items || [];



    const results =
        [];



    for (
        const item
        of items
    ) {


        const result =
            await processQueueItem(
                item
            );


        results.push(
            result
        );


    }



    return {


        success:
            true,


        processed:
            items.length,


        results


    };


}
