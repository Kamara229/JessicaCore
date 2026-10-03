/*
 * =========================================================
 * JESSICA LEARNING WORKER v3
 * =========================================================
 *
 * Обработчик Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Queue
 *        ↓
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 *        ↓
 * Learning Proposal
 *
 *
 * PATTERN_DISCOVERY
 *        ↓
 * Pattern Discovery Worker
 *        ↓
 * NEW_SKILL
 *        ↓
 * Learning Proposal
 *
 *
 * Ответственность:
 *
 * - получить ожидающие Queue Items;
 * - маршрутизировать Pattern Discovery;
 * - создать Proposal;
 * - сохранить Proposal;
 * - обновить Queue Status.
 *
 *
 * НЕ:
 *
 * - анализирует Experience самостоятельно;
 * - вызывает AI напрямую;
 * - принимает AUTO_APPROVE;
 * - создаёт Experience Skill;
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


import {
    processPatternDiscovery
} from "./patternDiscoveryWorker.js";





const VALID_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT",

    "PATTERN_DISCOVERY"

];





function validateQueueItem(
    item
) {


    if(
        !item ||
        typeof item !== "object"
    ){

        return false;

    }



    return VALID_ACTIONS.includes(
        item.action
    );

}





/*
 * =========================================================
 * MARK FAILED
 * =========================================================
 */


async function markFailed(
    item
) {


    try {


        if(
            item?.id
        ){

            await updateLearningQueueItemStatus(

                item.id,

                "FAILED"

            );

        }


    }catch(error){


        console.error(

            "Learning Queue FAILED update error:",

            error

        );

    }

}





/*
 * =========================================================
 * PREPARE ITEM
 * =========================================================
 */


async function prepareQueueItem(
    item
) {


    if(
        item.action !==
        "PATTERN_DISCOVERY"
    ){

        return {


            success:
                true,


            ignored:
                false,


            queueItem:
                item,


            discovery:
                null

        };

    }



    const discovery =

        await processPatternDiscovery(
            item
        );



    if(
        !discovery?.success
    ){

        return {


            success:
                false,


            ignored:
                false,


            queueItem:
                null,


            reason:

                discovery?.reason

                ||

                "Pattern Discovery failed",


            discovery

        };

    }



    if(
        discovery.ignored === true
    ){

        return {


            success:
                true,


            ignored:
                true,


            queueItem:
                null,


            reason:

                discovery.reason

                ||

                "Pattern Discovery не обнаружил reusable Experience",


            discovery

        };

    }



    if(
        !discovery.resolved
        ||
        !discovery.queueItem
    ){

        return {


            success:
                false,


            ignored:
                false,


            queueItem:
                null,


            reason:
                "Pattern Discovery не создал NEW_SKILL Queue Item",


            discovery

        };

    }



    return {


        success:
            true,


        ignored:
            false,


        queueItem:

            discovery.queueItem,


        discovery

    };

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


            success:
                false,


            queueItemId:

                item?.id ||

                null,


            reason:
                "Queue Item не поддерживается Learning Worker"

        };

    }



    try {


        /*
         * =================================================
         * 1. PREPARE
         * =================================================
         */


        const prepared =

            await prepareQueueItem(
                item
            );



        if(
            !prepared.success
        ){

            await markFailed(
                item
            );



            return {


                success:
                    false,


                queueItemId:
                    item.id,


                reason:
                    prepared.reason,


                discovery:
                    prepared.discovery ||

                    null

            };

        }



        /*
         * Pattern Extractor решил,
         * что reusable Experience нет.
         *
         * Это корректный terminal outcome.
         */


        if(
            prepared.ignored === true
        ){

            await updateLearningQueueItemStatus(

                item.id,

                "IGNORED"

            );



            return {


                success:
                    true,


                queueItemId:
                    item.id,


                ignored:
                    true,


                proposal:
                    null,


                reason:
                    prepared.reason,


                discovery:
                    prepared.discovery ||

                    null

            };

        }



        const effectiveItem =

            prepared.queueItem;



        /*
         * =================================================
         * 2. CREATE PROPOSAL
         * =================================================
         */


        const proposal =

            createLearningProposalFromQueue(
                effectiveItem
            );



        if(
            !proposal
        ){

            await markFailed(
                item
            );



            return {


                success:
                    false,


                queueItemId:
                    item.id,


                reason:
                    "Proposal не создан"

            };

        }



        /*
         * =================================================
         * 3. SAVE PROPOSAL
         * =================================================
         */


        const saved =

            await saveLearningProposal(
                proposal
            );



        if(
            !saved?.success
        ){

            await markFailed(
                item
            );



            return {


                success:
                    false,


                queueItemId:
                    item.id,


                reason:

                    saved?.error

                    ||

                    "Proposal не сохранён"

            };

        }



        /*
         * =================================================
         * 4. UPDATE QUEUE
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


                success:
                    false,


                proposal,


                queueItemId:
                    item.id,


                reason:
                    "Proposal создан, Queue не обновлена"

            };

        }



        /*
         * =================================================
         * 5. SUCCESS
         * =================================================
         */


        return {


            success:
                true,


            queueItemId:
                item.id,


            proposalId:
                proposal.id,


            action:
                proposal.action,


            proposal,


            discovery:

                prepared.discovery

                ||

                null

        };



    }catch(error){


        console.error(

            "Jessica Learning Worker error:",

            error

        );



        await markFailed(
            item
        );



        return {


            success:
                false,


            queueItemId:

                item?.id ||

                null,


            reason:

                error?.message

                ||

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



    if(
        !queueResult?.success
    ){

        return {


            success:
                false,


            processed:
                0,


            error:

                queueResult?.error

                ||

                "Ошибка получения Learning Queue"

        };

    }



    const items =

        Array.isArray(
            queueResult.items
        )

            ? queueResult.items

            : [];



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


        success:
            true,


        processed:

            items.length,


        successful:

            results.filter(

                item =>
                    item.success === true

            )
            .length,


        ignored:

            results.filter(

                item =>
                    item.ignored === true

            )
            .length,


        failed:

            results.filter(

                item =>
                    item.success === false

            )
            .length,


        results

    };

}
