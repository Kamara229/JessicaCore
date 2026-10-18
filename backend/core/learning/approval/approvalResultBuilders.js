/*
 * =========================================================
 * JESSICA APPROVAL RESULT BUILDERS
 * =========================================================
 */


import {
    persistProposalStatus
} from "./approvalPersistence.js";


import {
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


import {
    rejectCandidateMemory,
    keepCandidateMemoryActive
} from "./approvalCandidateMemory.js";


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


export async function buildRejectedResult({

    proposal,

    reason,

    review = null,

    quality = null

}) {

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

        reinforced:
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


export async function buildKeepCandidateResult({

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


    const candidateMemory =

        keepCandidateMemoryActive(
            proposal
        );


    return {

        success:
            true,

        learned:
            false,

        reinforced:
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


export async function buildFailedResult({

    proposal,

    reason,

    review = null,

    quality = null,

    autonomy = null,

    sourceResult = null

}) {

    const persistence =

        proposal?.id

            ? await persistStatus(

                proposal,

                APPROVAL_RESULT_STATUS.FAILED

            )

            : {

                success:
                    false,

                error:
                    "Proposal ID отсутствует"

            };


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

        reinforced:
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


export async function markProposalApproved(
    proposal
) {

    return await persistStatus(

        proposal,

        APPROVAL_RESULT_STATUS.APPROVED

    );

}
