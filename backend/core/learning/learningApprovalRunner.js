/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER v7
 * =========================================================
 *
 * Финальный координатор
 * автономного Learning Decision.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Reviewer
 *        ↓
 * Quality Gate
 *        ↓
 * Autonomy Policy
 *        ↓
 *
 * ┌──────────────────────────────────────────────┐
 * │                                              │
 * ↓                                              ↓
 *
 * KEEP_CANDIDATE                            AUTO_APPROVE
 *      ↓                                         ↓
 * Proposal Status                           Save Skill
 *      ↓                                         ↓
 * Candidate ACTIVE                          Proposal APPROVED
 *                                                ↓
 *                                         Candidate PROMOTED
 *
 *
 * Structural / Quality Reject
 *        ↓
 * Proposal REJECTED
 *        ↓
 * Candidate REJECTED
 *
 *
 * Technical Failure
 *        ↓
 * Proposal FAILED
 *        ↓
 * Candidate remains ACTIVE
 *
 *
 * НЕ:
 *
 * - анализирует Execution Trace;
 * - строит Experience Skill;
 * - объединяет Candidates;
 * - вызывает AI;
 * - работает с Supabase напрямую.
 *
 * =========================================================
 */


import {
    reviewLearningProposal
} from "./learningReviewer.js";


import {
    validateLearningQuality
} from "./learningQualityGate.js";


import {
    evaluateLearningAutonomy
} from "./learningAutonomyPolicy.js";


import {
    approveAndSaveLearningProposal
} from "../../experience/learning/learningApproval.js";


import {
    persistProposalStatus
} from "./approval/approvalPersistence.js";


import {
    APPROVAL_ACTION,
    APPROVAL_RESULT_STATUS
} from "./approval/approvalConstants.js";


import {
    promoteCandidateMemory,
    rejectCandidateMemory,
    keepCandidateMemoryActive
} from "./approval/approvalCandidateMemory.js";





/*
 * =========================================================
 * PROPOSAL STATUS
 * =========================================================
 */


async function persistStatus(
    proposal,
    status
) {

    const result =

        await persistProposalStatus({

            proposal,

            status

        });


    return {

        success:
            result?.success === true,

        error:

            result?.success === true

                ? null

                : (
                    result?.error

                    ||

                    "Proposal status не сохранён"
                )

    };

}





/*
 * =========================================================
 * REJECT RESULT
 * =========================================================
 */


async function buildRejectedResult({

    proposal,

    reason,

    review = null,

    quality = null

}) {


    /*
     * Proposal и Candidate Memory
     * являются независимыми persistence
     * операциями.
     *
     * Ошибка одной не должна скрывать
     * результат другой.
     */


    const persistence =

        await persistStatus(

            proposal,

            APPROVAL_RESULT_STATUS.REJECTED

        );


    const candidateMemory =

        await rejectCandidateMemory({

            proposal,

            reason

        });


    return {

        success:
            true,

        learned:
            false,

        status:
            APPROVAL_RESULT_STATUS.REJECTED,

        reason,

        review,

        quality,

        persistence,

        candidateMemory

    };

}





/*
 * =========================================================
 * KEEP CANDIDATE RESULT
 * =========================================================
 */


async function buildKeepCandidateResult({

    proposal,

    reason,

    review,

    quality,

    autonomy

}) {


    const persistence =

        await persistStatus(

            proposal,

            APPROVAL_RESULT_STATUS.KEEP_CANDIDATE

        );


    /*
     * ACTIVE уже сохранён
     * Candidate Memory Storage.
     *
     * Здесь только фиксируем semantic
     * состояние в результате Runner.
     */


    const candidateMemory =

        keepCandidateMemoryActive(
            proposal
        );


    return {

        success:
            true,

        learned:
            false,

        status:
            APPROVAL_RESULT_STATUS.KEEP_CANDIDATE,

        reason,

        review,

        quality,

        autonomy,

        persistence,

        candidateMemory

    };

}





/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


async function buildFailedResult({

    proposal,

    reason,

    review = null,

    quality = null,

    autonomy = null,

    sourceResult = null

}) {


    const persistence =

        proposal?.id

            ?

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.FAILED

            )

            :

            {

                success:
                    false,

                error:
                    "Proposal ID отсутствует"

            };


    /*
     * Техническая ошибка НЕ должна
     * переводить Candidate в REJECTED.
     *
     * Он остаётся ACTIVE и может быть
     * использован повторно.
     */


    const candidateMemory =

        proposal

            ? keepCandidateMemoryActive(
                proposal
            )

            : {

                success:
                    true,

                updated:
                    false,

                skipped:
                    true,

                reason:
                    "Proposal отсутствует"

            };


    return {

        ...(sourceResult || {}),

        success:
            false,

        learned:
            false,

        status:
            APPROVAL_RESULT_STATUS.FAILED,

        reason,

        review,

        quality,

        autonomy,

        persistence,

        candidateMemory

    };

}





/*
 * =========================================================
 * PROCESS SINGLE PROPOSAL
 * =========================================================
 */


async function processProposal(
    proposal
) {


    /*
     * =====================================================
     * 0. INPUT
     * =====================================================
     */


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return {

            success:
                false,

            learned:
                false,

            status:
                APPROVAL_RESULT_STATUS.FAILED,

            reason:
                "Proposal отсутствует"

        };

    }



    /*
     * =====================================================
     * 1. REVIEW
     * =====================================================
     */


    const review =

        reviewLearningProposal(
            proposal
        );


    if(
        review.valid !== true
    ){

        return buildRejectedResult({

            proposal,

            reason:

                review.reason

                ||

                "Proposal не прошёл Reviewer",

            review

        });

    }



    /*
     * =====================================================
     * 2. QUALITY GATE
     * =====================================================
     */


    const quality =

        validateLearningQuality(
            proposal
        );


    if(
        quality.passed !== true
    ){

        return buildRejectedResult({

            proposal,

            reason:

                quality.reason

                ||

                "Proposal не прошёл Quality Gate",

            review,

            quality

        });

    }



    /*
     * =====================================================
     * 3. AUTONOMY POLICY
     * =====================================================
     */


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );


    if(
        autonomy.action !==
        APPROVAL_ACTION.AUTO_APPROVE
    ){

        return buildKeepCandidateResult({

            proposal,

            reason:

                autonomy.reason

                ||

                "Недостаточно Evidence для AUTO_APPROVE",

            review,

            quality,

            autonomy

        });

    }



    /*
     * =====================================================
     * 4. SAVE EXPERIENCE
     * =====================================================
     */


    let result;


    try {


        result =

            await approveAndSaveLearningProposal({

                proposal,

                autonomy,

                confidence:

                    autonomy.confidence

                    ??

                    proposal.confidence

                    ??

                    null

            });


    }catch(error){


        console.error(

            "Jessica Learning Approval Runner error:",

            error

        );


        return buildFailedResult({

            proposal,

            reason:

                error?.message

                ||

                "Learning Approval failed",

            review,

            quality,

            autonomy

        });

    }



    /*
     * =====================================================
     * 5. EXPERIENCE SAVE FAILED
     * =====================================================
     */


    if(
        result?.success !== true
    ){

        return buildFailedResult({

            proposal,

            reason:

                result?.error

                ||

                result?.reason

                ||

                "Experience Skill не сохранён",

            review,

            quality,

            autonomy,

            sourceResult:
                result

        });

    }



    /*
     * =====================================================
     * 6. PROPOSAL → APPROVED
     * =====================================================
     */


    const persistence =

        await persistStatus(

            proposal,

            APPROVAL_RESULT_STATUS.APPROVED

        );



    /*
     * =====================================================
     * 7. CANDIDATE → PROMOTED
     * =====================================================
     *
     * Skill уже физически сохранён.
     *
     * Если здесь произойдёт ошибка,
     * нельзя говорить, что Learning
     * не состоялся.
     *
     * Ошибка будет явно возвращена
     * как Candidate Memory
     * finalization problem.
     *
     * =====================================================
     */


    const candidateMemory =

        await promoteCandidateMemory({

            proposal,

            skillId:

                result.skillId

                ||

                result
                    ?.experience
                    ?.id

                ||

                null

        });



    /*
     * =====================================================
     * 8. SUCCESS
     * =====================================================
     */


    return {

        ...result,

        success:
            true,

        learned:
            true,

        status:
            APPROVAL_RESULT_STATUS.APPROVED,

        proposalStateUpdated:

            persistence.success,

        proposalStateError:

            persistence.error,

        candidateMemoryStateUpdated:

            candidateMemory.success,

        candidateMemoryStateError:

            candidateMemory.success

                ? null

                : candidateMemory.error,

        review,

        quality,

        autonomy,

        persistence,

        candidateMemory

    };

}





/*
 * =========================================================
 * RUN BATCH
 * =========================================================
 */


export async function runLearningApproval(
    proposals = []
) {


    if(
        !Array.isArray(
            proposals
        )
    ){

        return {

            success:
                false,

            processed:
                0,

            learned:
                0,

            candidates:
                0,

            rejected:
                0,

            failed:
                0,

            error:
                "Invalid proposals"

        };

    }


    const results = [];


    for(
        const proposal
        of proposals
    ){

        const result =

            await processProposal(
                proposal
            );


        results.push(
            result
        );

    }


    return {

        success:
            true,

        processed:
            proposals.length,


        /*
         * Skill действительно сохранён.
         */


        learned:

            results.filter(

                item =>
                    item.learned === true

            )
            .length,


        /*
         * Candidate продолжает
         * накапливать Evidence.
         */


        candidates:

            results.filter(

                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.KEEP_CANDIDATE

            )
            .length,


        rejected:

            results.filter(

                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.REJECTED

            )
            .length,


        failed:

            results.filter(

                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.FAILED

            )
            .length,


        /*
         * Skill сохранён, но возникла
         * проблема финализации
         * Candidate Memory.
         */


        candidateMemoryErrors:

            results.filter(

                item =>

                    item.learned === true

                    &&

                    item.candidateMemoryStateUpdated === false

                    &&

                    item
                        ?.candidateMemory
                        ?.skipped !== true

            )
            .length,


        results

    };

}
