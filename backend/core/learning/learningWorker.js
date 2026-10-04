/*
 * =========================================================
 * JESSICA LEARNING WORKER v4
 * =========================================================
 *
 * Обработчик Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Queue
 *        ↓
 * Prepare Queue Item
 *        ↓
 *
 * PATTERN_DISCOVERY
 *        ↓
 * Pattern Discovery Worker
 *        ↓
 * NEW_SKILL
 *
 *
 * NEW_SKILL
 *        ↓
 * Candidate Memory
 *        ↓
 * create / exact match / similarity match
 *        ↓
 * merge evidence
 *        ↓
 * recalculated Candidate
 *
 *
 * SKILL_IMPROVEMENT
 *        ↓
 * Candidate Memory v1 не применяется
 *
 *
 *        ↓
 * Learning Proposal
 *        ↓
 * Proposal Storage
 *        ↓
 * Queue → PROPOSED
 *
 *
 * Ответственность:
 *
 * - получить Pending Queue Items;
 * - выполнить Pattern Discovery;
 * - накопить NEW_SKILL Candidate Memory;
 * - создать Proposal;
 * - сохранить Proposal;
 * - обновить Queue Status.
 *
 *
 * НЕ:
 *
 * - анализирует Execution Trace;
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


import {
    observeLearningCandidate
} from "./learningCandidateMemory.js";





const VALID_ACTIONS = [

    "NEW_SKILL",

    "SKILL_IMPROVEMENT",

    "PATTERN_DISCOVERY"

];





/*
 * =========================================================
 * VALIDATE
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
 * PATTERN DISCOVERY
 * =========================================================
 *
 * PATTERN_DISCOVERY должен быть
 * преобразован в обычный NEW_SKILL
 * до Candidate Memory.
 *
 * =========================================================
 */


async function prepareDiscovery(
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


    /*
     * Pattern Extractor может корректно
     * решить, что reusable Skill
     * из данного Execution извлечь нельзя.
     *
     * Это не техническая ошибка.
     */


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
 * CANDIDATE MEMORY
 * =========================================================
 *
 * Candidate Memory v1 применяется
 * только к NEW_SKILL.
 *
 *
 * Первый Execution:
 *
 * candidate
 *      ↓
 * ACTIVE Memory
 *
 *
 * Следующий похожий Execution:
 *
 * ACTIVE Memory
 *      +
 * incoming candidate
 *      ↓
 * merged candidate
 *      ↓
 * occurrences / confidence / maturity ↑
 *
 * =========================================================
 */


async function prepareCandidateMemory(
    queueItem
) {

    if(
        queueItem?.action !==
        "NEW_SKILL"
    ){

        return {

            success:
                true,

            observed:
                false,

            queueItem,

            memory:
                null,

            candidate:
                null

        };

    }


    const observation =

        await observeLearningCandidate(
            queueItem
        );


    if(
        !observation?.success
    ){

        return {

            success:
                false,

            observed:
                false,

            queueItem:
                null,

            memory:
                observation?.memory || null,

            candidate:
                observation?.candidate || null,

            reason:

                observation?.error

                ||

                "Candidate Memory observation failed",

            observation

        };

    }


    if(
        !observation.queueItem
    ){

        return {

            success:
                false,

            observed:
                observation.observed === true,

            queueItem:
                null,

            memory:
                observation.memory || null,

            candidate:
                observation.candidate || null,

            reason:
                "Candidate Memory не вернул Queue Item",

            observation

        };

    }


    return {

        success:
            true,

        observed:
            observation.observed === true,

        created:
            observation.created === true,

        merged:
            observation.merged === true,

        similarity:

            observation.similarity

            ??

            null,

        queueItem:
            observation.queueItem,

        memory:
            observation.memory || null,

        candidate:
            observation.candidate || null,

        observation

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

                item?.id

                ||

                null,

            reason:
                "Queue Item не поддерживается Learning Worker"

        };

    }


    try {


        /*
         * =================================================
         * 1. PATTERN DISCOVERY
         * =================================================
         */


        const discoveryResult =

            await prepareDiscovery(
                item
            );


        if(
            !discoveryResult.success
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
                    discoveryResult.reason,

                discovery:
                    discoveryResult.discovery || null

            };

        }


        /*
         * =================================================
         * 2. DISCOVERY IGNORED
         * =================================================
         */


        if(
            discoveryResult.ignored === true
        ){

            const updated =

                await updateLearningQueueItemStatus(

                    item.id,

                    "IGNORED"

                );


            return {

                success:
                    updated?.success === true,

                queueItemId:
                    item.id,

                ignored:
                    true,

                proposal:
                    null,

                reason:
                    discoveryResult.reason,

                discovery:
                    discoveryResult.discovery || null,

                queueUpdated:
                    updated?.success === true,

                queueUpdateError:

                    updated?.success === true

                        ? null

                        : (
                            updated?.error ||
                            "Queue status не обновлён"
                        )

            };

        }


        /*
         * После Pattern Discovery
         * здесь уже может быть NEW_SKILL.
         */


        let effectiveItem =

            discoveryResult.queueItem;



        /*
         * =================================================
         * 3. CANDIDATE MEMORY
         * =================================================
         */


        const memoryResult =

            await prepareCandidateMemory(
                effectiveItem
            );


        if(
            !memoryResult.success
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
                    memoryResult.reason,

                discovery:
                    discoveryResult.discovery || null,

                candidateMemory:
                    memoryResult.observation || null

            };

        }


        /*
         * КРИТИЧНО:
         *
         * Proposal должен создаваться уже
         * из Queue Item, обогащённого
         * накопленным Candidate.
         */


        effectiveItem =

            memoryResult.queueItem;



        /*
         * =================================================
         * 4. CREATE PROPOSAL
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
                    "Proposal не создан",

                candidateMemory:
                    memoryResult.observation || null

            };

        }



        /*
         * =================================================
         * 5. SAVE PROPOSAL
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

                    "Proposal не сохранён",

                proposal,

                candidateMemory:
                    memoryResult.observation || null

            };

        }



        /*
         * =================================================
         * 6. UPDATE QUEUE
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
                    "Proposal создан, Queue не обновлена",

                discovery:
                    discoveryResult.discovery || null,

                candidateMemory:
                    memoryResult.observation || null

            };

        }



        /*
         * =================================================
         * 7. SUCCESS
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
                discoveryResult.discovery || null,

            candidateMemory:

                memoryResult.observed === true

                    ? {

                        id:

                            memoryResult.memory?.id

                            ||

                            null,

                        created:

                            memoryResult.created === true,

                        merged:

                            memoryResult.merged === true,

                        similarity:

                            memoryResult.similarity,

                        occurrences:

                            memoryResult
                                ?.candidate
                                ?.occurrences

                            ??

                            null,

                        confidence:

                            memoryResult
                                ?.candidate
                                ?.confidence

                            ??

                            null

                    }

                    : null

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

                item?.id

                ||

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

            successful:
                0,

            ignored:
                0,

            failed:
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
