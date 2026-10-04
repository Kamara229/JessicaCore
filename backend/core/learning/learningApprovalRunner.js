/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER v6
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
 * ┌───────────────────┬───────────────────┐
 * ↓                   ↓                   ↓
 * REJECT          KEEP_CANDIDATE      AUTO_APPROVE
 * ↓                   ↓                   ↓
 * Proposal Status     Proposal Status     Save Skill
 *                                         ↓
 *                                     Proposal Status
 *
 *
 * НЕ:
 *
 * - анализирует Execution Trace;
 * - строит Skill;
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
                    result?.error ||
                    "Proposal status не сохранён"
                )

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

        const persistence =

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.REJECTED

            );


        return {

            success:
                true,

            learned:
                false,

            status:
                APPROVAL_RESULT_STATUS.REJECTED,

            reason:
                review.reason,

            review,

            persistence

        };

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

        const persistence =

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.REJECTED

            );


        return {

            success:
                true,

            learned:
                false,

            status:
                APPROVAL_RESULT_STATUS.REJECTED,

            reason:
                quality.reason,

            review,

            quality,

            persistence

        };

    }


    /*
     * =====================================================
     * 3. AUTONOMY
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

        const persistence =

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.KEEP_CANDIDATE

            );


        return {

            success:
                true,

            learned:
                false,

            status:
                APPROVAL_RESULT_STATUS.KEEP_CANDIDATE,

            reason:
                autonomy.reason,

            review,

            quality,

            autonomy,

            persistence

        };

    }


    /*
     * =====================================================
     * 4. SAVE EXPERIENCE
     * =====================================================
     */


    try {


        const result =

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


        if(
            result?.success !== true
        ){

            const persistence =

                await persistStatus(

                    proposal,

                    APPROVAL_RESULT_STATUS.FAILED

                );


            return {

                ...result,

                success:
                    false,

                learned:
                    false,

                status:
                    APPROVAL_RESULT_STATUS.FAILED,

                review,

                quality,

                autonomy,

                persistence

            };

        }


        /*
         * Skill уже сохранён.
         *
         * Теперь Proposal должен
         * получить финальный статус
         * в persistent storage.
         */


        const persistence =

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.APPROVED

            );


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

            review,

            quality,

            autonomy,

            persistence

        };


    }catch(error){


        console.error(

            "Jessica Learning Approval Runner error:",

            error

        );


        const persistence =

            await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.FAILED

            );


        return {

            success:
                false,

            learned:
                false,

            status:
                APPROVAL_RESULT_STATUS.FAILED,

            reason:

                error?.message

                ||

                "Learning Approval failed",

            review,

            quality,

            autonomy,

            persistence

        };

    }

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

        learned:

            results.filter(

                item =>
                    item.learned === true

            )
            .length,

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

        results

    };

}
