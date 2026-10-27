/*
 * =========================================================
 * JESSICA LEARNING WORKER v5
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
    updateLearningQueueItemStatus
} from "./learningQueueStorage.js";
import {claimPendingLearningItems} from "./learningQueueClaim.js";


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

import {
    getLatestExperienceVersion
} from "../../experience/storage/experienceStorage.js";
import {checkLearningWorkerReplay} from "./workerReplayGuard.js";
import {recordProposalSaveFailure} from "./proposalSaveRetry.js";
import {saveLearningProposalOwned} from "./proposalStorage/proposalOwnedWriter.js";





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
    item,
    updateStatus=updateLearningQueueItemStatus
) {

    try {


        if(
            item?.id
        ){

            await updateStatus(

                item.id,

                "FAILED",

                item.claimToken || null

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

            retryLater:
                discovery?.retryLater === true,

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


export async function processQueueItem(
    item,
    {
      replayGuard=checkLearningWorkerReplay,
      prepareMemory=prepareCandidateMemory,
      proposalSaver=saveLearningProposal,
      proposalFailureHandler=recordProposalSaveFailure,
      updateStatus=updateLearningQueueItemStatus
    }={}
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

        // Never reobserve Candidate Memory or create a second Proposal for
        // the same Queue Item after an interrupted/duplicated Worker pass.
        const replay=await replayGuard(item);
        if(replay?.handled===true)return replay.result;

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

            /*
             * Provider quota / temporary AI unavailability
             * не является Learning failure.
             *
             * Queue Item остаётся PENDING и будет
             * повторно обработан следующим daemon cycle.
             */


            if(
                discoveryResult.retryLater === true
            ){

                console.log(

                    "Jessica Learning deferred:",

                    {

                        queueItemId:
                            item.id,

                        action:
                            item.action,

                        reason:
                            discoveryResult.reason

                    }

                );


                return {

                    success:
                        true,

                    deferred:
                        true,

                    queueItemId:
                        item.id,

                    reason:
                        discoveryResult.reason,

                    discovery:
                        discoveryResult.discovery || null

                };

            }


            await markFailed(
                item,updateStatus
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

                await updateStatus(

                    item.id,

                    "IGNORED",
                    item.claimToken || null

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
         * NEW_SKILL IDENTITY GUARD
         * =================================================
         * An un-used existing Skill must not be overwritten
         * by a generic seed Pattern sharing the same id.
         * Existing Skills evolve through SKILL_IMPROVEMENT
         * only after genuinely using that Skill.
         */
        if (effectiveItem?.action === "NEW_SKILL") {
            const candidate =
                effectiveItem?.event?.payload?.skillCandidate
                || effectiveItem?.event_json?.payload?.skillCandidate;
            const skillId = String(candidate?.skillId || "").trim();

            if (skillId) {
                let existing = null;
                try {
                    existing = await getLatestExperienceVersion(skillId);
                } catch (error) {
                    console.warn("Jessica identity guard deferred:", error?.message);
                    return {
                        success: true,
                        deferred: true,
                        queueItemId: item.id,
                        reason: "Не удалось проверить существующую версию Skill"
                    };
                }

                if (existing) {
                    console.warn("Jessica duplicate NEW_SKILL rejected:", skillId);
                    const updated = await updateStatus(item.id, "IGNORED", item.claimToken || null);
                    return {
                        success: updated?.success === true,
                        ignored: true,
                        queueItemId: item.id,
                        reason: "NEW_SKILL совпадает с существующим Skill ID; переобучение без применения запрещено",
                        queueUpdated: updated?.success === true
                    };
                }
            }
        }

        /*
         * =================================================
         * 3. CANDIDATE MEMORY
         * =================================================
         */


        const memoryResult =

            await prepareMemory(
                effectiveItem
            );


        if(
            !memoryResult.success
        ){

            await markFailed(
                item,updateStatus
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
                item,updateStatus
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


        let saved;
        try {
            saved=item.claimToken && proposalSaver===saveLearningProposal
              ? await saveLearningProposalOwned(proposal,{claimToken:item.claimToken})
              : await proposalSaver(proposal);
        } catch(error){
            saved={success:false,error:error?.message||"Proposal storage error"};
        }

        if(saved?.success!==true){
            // Candidate observation is already committed. The dedicated SQL
            // retry facade performs a journal check, counter increment and
            // Queue release in one atomic transaction.
            if(effectiveItem.action==="NEW_SKILL"){
                const retry=await proposalFailureHandler({queueItemId:item.id,claimToken:item.claimToken||null});
                if(retry.status==="already_saved"){
                    return {
                        success:true,existing:true,queueUpdated:true,
                        queueItemId:item.id,proposalId:retry.proposalId,
                        action:proposal.action,recovered:true
                    };
                }
                if(retry.status==="retry"){
                    return {
                        success:true,deferred:true,queueAlreadyReleased:true,
                        retryScheduled:true,attempts:retry.attempts,
                        queueItemId:item.id,reason:saved?.error||"Proposal save deferred"
                    };
                }
                if(retry.status==="exhausted"){
                    return {
                        success:false,exhausted:true,attempts:retry.attempts,
                        queueItemId:item.id,reason:"Proposal retry limit exceeded"
                    };
                }
                // Missing journal, permission error, or RPC outage:
                // leave the claimed row untouched for conservative recovery.
                return {
                    success:false,queueItemId:item.id,
                    recoveryBlocked:true,
                    reason:retry.error||"Proposal retry unavailable"
                };
            }
            // SKILL_IMPROVEMENT does not mutate Candidate Memory.
            await markFailed(item,updateStatus);
            return {
                success:false,queueItemId:item.id,
                reason:saved?.error||"Proposal не сохранён"
            };
        }

        /*
         * =================================================
         * 6. UPDATE QUEUE
         * =================================================
         */


        const updated =

            await updateStatus(

                item.id,

                "PROPOSED",
                item.claimToken || null

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
                saved.proposal?.id || proposal.id,

            existing:
                saved.existing === true,

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
            item,updateStatus
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


export async function runLearningWorker({
    claim=claimPendingLearningItems,
    processItem=processQueueItem,
    updateQueue=updateLearningQueueItemStatus
}={})
{

    const queueResult =

        await claim();


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

            await processItem(
                item
            );


        // Deferred provider/lookup results must relinquish their claim.
        // Never leave an item PROCESSING after a voluntary retry deferral.
        if(result?.deferred===true && result?.queueAlreadyReleased!==true){
            const returned=await updateQueue(item.id,"PENDING",item.claimToken||null);
            if(returned?.success!==true){
                result.success=false;
                result.reason=returned?.error||"Cannot release deferred queue claim";
            }
        }
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

        deferred:

            results.filter(

                item =>
                    item.deferred === true

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
