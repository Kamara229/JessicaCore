/*
 * =========================================================
 * JESSICA LEARNING WORKER v2
 * =========================================================
 *
 * Обработчик Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Queue
 *        ↓
 * Learning Worker
 *        ↓
 * Learning Proposal
 *        ↓
 * Autonomy Policy
 *        ↓
 * Approval Runner
 *
 *
 * Ответственность:
 *
 * - получить ожидающие события;
 * - проверить возможность обучения;
 * - создать Proposal;
 * - сохранить Proposal;
 * - изменить статус Queue.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - принимает решение обучения;
 * - создаёт Skill;
 * - сохраняет Experience.
 *
 * =========================================================
 */



import {
    getPendingLearningItems,
    updateLearningQueueItemStatus
} from "./learningQueueStorage.js";


import {
    createLearningProposalFromQueue
} from "../../experience/learning/learningProposal.js";


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


    if(
        !item ||
        typeof item !== "object"
    ){

        return false;

    }



    const allowedActions = [

        "NEW_SKILL",

        "SKILL_IMPROVEMENT"

    ];



    return allowedActions.includes(
        item.action
    );

}









/*
 * =========================================================
 * PROCESS ITEM
 * =========================================================
 */


async function processQueueItem(
    item
) {


    if(
        !validateQueueItem(
            item
        )
    ){

        return {


            success:false,


            queueItemId:
                item?.id || null,


            reason:
                "Queue item не требует обучения"


        };

    }







    try {





        /*
         * =================================================
         * CREATE PROPOSAL
         * =================================================
         */


        const proposal =

            createLearningProposalFromQueue(
                item
            );






        if(
            !proposal
        ){


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
         * SAVE PROPOSAL
         * =================================================
         */


        const saved =

            await saveLearningProposal(
                proposal
            );






        if(
            !saved?.success
        ){


            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );



            return {


                success:false,


                queueItemId:
                    item.id,


                reason:

                    saved?.error ||

                    "Proposal не сохранён"


            };

        }









        /*
         * =================================================
         * UPDATE QUEUE
         * =================================================
         */


        const updated =

            await updateLearningQueueItemStatus(

                item.id,

                "PROPOSED"

            );






        if(
            !updated?.success
        ){


            return {


                success:false,


                proposal,


                queueItemId:
                    item.id,


                reason:
                    "Proposal создан, Queue не обновлена"


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






    } catch(error){



        console.error(

            "Jessica Learning Worker error:",

            error

        );



        try{


            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );


        }catch(updateError){


            console.error(

                "Learning Queue update error:",

                updateError

            );


        }





        return {


            success:false,


            queueItemId:
                item?.id || null,


            reason:
                error.message || "Worker error"


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





    if(
        !queueResult?.success
    ){

        return {


            success:false,


            processed:0,


            error:

                queueResult?.error ||

                "Ошибка получения Learning Queue"


        };

    }






    const items =

        Array.isArray(
            queueResult.items
        )

        ?

        queueResult.items

        :

        [];








    const results = [];





    for(
        const item
        of items
    ){

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
