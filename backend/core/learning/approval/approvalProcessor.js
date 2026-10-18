/*
 * =========================================================
 * JESSICA APPROVAL PROCESSOR FACADE
 * =========================================================
 *
 * Один Proposal:
 * evaluation → persistence route.
 * =========================================================
 */


import {
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


import {
    APPROVAL_FLOW,
    evaluateApprovalProposal
} from "./approvalEvaluation.js";


import {
    buildRejectedResult,
    buildKeepCandidateResult
} from "./approvalResultBuilders.js";


import {
    persistSkillReinforcement
} from "./reinforcementPersistence.js";


import {
    saveApprovedSkillVersion
} from "./approvalSkillSave.js";


export async function processLearningProposal(
    proposal
) {

    if(
        !proposal
        ||
        typeof proposal !== "object"
    ){

        return {

            success:
                false,

            learned:
                false,

            reinforced:
                false,

            status:
                APPROVAL_RESULT_STATUS.FAILED,

            reason:
                "Proposal отсутствует"

        };

    }


    const evaluation =

        evaluateApprovalProposal(
            proposal
        );


    if(
        evaluation.flow ===
        APPROVAL_FLOW.REJECT
    ){

        return await buildRejectedResult({

            proposal,

            reason:
                evaluation.reason,

            review:
                evaluation.review,

            quality:
                evaluation.quality

        });

    }


    if(
        evaluation.flow ===
        APPROVAL_FLOW.KEEP
    ){

        return await buildKeepCandidateResult({

            proposal,

            reason:
                evaluation.reason,

            review:
                evaluation.review,

            quality:
                evaluation.quality,

            autonomy:
                evaluation.autonomy

        });

    }


    if(
        evaluation.flow ===
        APPROVAL_FLOW.REINFORCE
    ){

        return await persistSkillReinforcement({

            proposal,

            quality:
                evaluation.quality,

            autonomy:
                evaluation.autonomy

        });

    }


    return await saveApprovedSkillVersion({

        proposal,

        review:
            evaluation.review,

        quality:
            evaluation.quality,

        autonomy:
            evaluation.autonomy

    });

}
