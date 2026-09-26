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
 * Learning Proposal Builder
 *        ↓
 * learning_proposals
 *        ↓
 * Learning Approval
 *
 *
 * Ответственность:
 *
 * - получить PENDING Queue Items;
 * - создать Learning Proposal;
 * - сохранить Proposal;
 * - обновить статус очереди.
 *
 *
 * НЕ:
 *
 * - создаёт Experience Skill;
 * - сохраняет Experience;
 * - принимает Approval решение.
 *
 * =========================================================
 */


import {
    getPendingLearningItems,
    updateLearningQueueItemStatus
} from "./learningQueueStorage.js";


import {
    createLearningProposalFromQueue
} from "./learningProposal.js";


import {
    saveLearningProposal
} from "./learningProposalStorage.js";





/*
 * =========================================================
 * PROCESS SINGLE QUEUE ITEM
 * =========================================================
 */


async function processQueueItem(
    item
) {


    if (
        !item ||
        typeof item !== "object"
    ) {


        return {

            success:
                false,

            reason:
                "Empty queue item"

        };

    }





    try {


        /*
         * =================================================
         * 1. CREATE PROPOSAL
         * =================================================
         */


        const proposal =
            createLearningProposalFromQueue(
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

                queueItemId:
                    item.id,

                reason:
                    "Learning Proposal не создан"

            };

        }





        /*
         * =================================================
         * 2. SAVE PROPOSAL
         * =================================================
         */


        const saveResult =
            await saveLearningProposal(
                proposal
            );





        if (
            !saveResult?.success
        ) {


            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );


            return {

                success:
                    false,

                queueItemId:
                    item.id,

                proposal,

                reason:
                    saveResult?.error ||
                    "Proposal не сохранён"

            };

        }





        /*
         * =================================================
         * 3. UPDATE QUEUE STATUS
         * =================================================
         */


        const updateResult =
            await updateLearningQueueItemStatus(

                item.id,

                "PROPOSED"

            );





        if (
            !updateResult?.success
        ) {


            return {

                success:
                    false,

                proposal,

                reason:
                    "Proposal сохранён, но очередь не обновлена"

            };

        }





        return {


            success:
                true,


            queueItemId:
                item.id,


            proposalId:
                proposal.id,


            proposal



        };



    } catch(error) {


        console.error(

            "Jessica Learning Worker error:",

            error

        );



        try {


            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );


        } catch(updateError) {


            console.error(

                "Learning Queue status update error:",

                updateError

            );


        }





        return {


            success:
                false,


            queueItemId:
                item.id,


            reason:
                error?.message ||
                "Worker error"


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


    /*
     * =====================================================
     * GET QUEUE
     * =====================================================
     */


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
                queueResult?.error ||
                "Не удалось получить Learning Queue"


        };

    }





    const items =
        Array.isArray(
            queueResult.items
        )
            ? queueResult.items
            : [];





    const results =
        [];





    /*
     * =====================================================
     * PROCESS
     * =====================================================
     */


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


        successful:
            results.filter(
                item =>
                    item.success === true
            ).length,


        failed:
            results.filter(
                item =>
                    item.success === false
            ).length,


        results



    };


}
