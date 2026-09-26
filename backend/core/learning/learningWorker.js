/*
 * =========================================================
 * JESSICA LEARNING WORKER
 * =========================================================
 *
 * Обработчик Learning Queue.
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
 * Approval
 *
 *
 * Ответственность:
 *
 * - получить ожидающие события;
 * - создать Proposal;
 * - сохранить Proposal;
 * - изменить статус Queue.
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - сохраняет Experience;
 * - выполняет Approval.
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
 * VALID QUEUE ITEM
 * =========================================================
 */


function validateQueueItem(
    item
) {


    if (
        !item ||
        typeof item !== "object"
    ) {

        return false;

    }


    if (
        !item.action
    ) {

        return false;

    }


    return true;

}









/*
 * =========================================================
 * PROCESS ITEM
 * =========================================================
 */


async function processQueueItem(
    item
) {


    if (
        !validateQueueItem(
            item
        )
    ) {


        return {

            success:false,

            reason:
                "Некорректный Learning Queue Item"

        };

    }







    try {



        /*
         * =================================================
         * 1. BUILD PROPOSAL
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

                success:false,

                queueItemId:
                    item.id,

                reason:
                    "Proposal не создан"

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


                success:false,


                queueItemId:
                    item.id,


                reason:

                    saveResult?.error ||

                    "Proposal сохранение не подтверждено"


            };


        }









        /*
         * =================================================
         * 3. UPDATE QUEUE
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


                success:false,


                proposal,


                reason:

                    "Proposal создан, но Queue не обновлена"


            };

        }









        return {


            success:true,


            queueItemId:

                item.id,


            proposalId:

                proposal.id,


            action:

                proposal.action,


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

                "Learning Queue update error:",

                updateError

            );

        }






        return {


            success:false,


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


    const queueResult =
        await getPendingLearningItems();




    if (
        !queueResult?.success
    ) {


        return {


            success:false,


            processed:0,


            error:

                queueResult?.error ||

                "Ошибка получения очереди"


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


        success:true,


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
