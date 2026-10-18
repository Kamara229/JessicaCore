/*
 * =========================================================
 * JESSICA APPROVAL PROCESSOR
 * =========================================================
 *
 * Один Proposal:
 * Reviewer → Quality → Autonomy → Persistence.
 * =========================================================
 */


import {
    reviewLearningProposal
} from "../learningReviewer.js";


import {
    validateLearningQuality
} from "../learningQualityGate.js";


import {
    evaluateLearningAutonomy
} from "../learningAutonomyPolicy.js";


import {
    approveAndSaveLearningProposal
} from "../../../experience/learning/learningApproval.js";


import {
    APPROVAL_ACTION,
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


import {
    promoteCandidateMemory
} from "./approvalCandidateMemory.js";


import {
    persistSkillReinforcement
} from "./reinforcementPersistence.js";


import {
    buildRejectedResult,
    buildKeepCandidateResult,
    buildFailedResult,
    markProposalApproved
} from "./approvalResultBuilders.js";


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


    const review =

        reviewLearningProposal(
            proposal
        );


    if(
        review.valid !== true
    ){

        return await buildRejectedResult({

            proposal,

            reason:

                review.reason

                ||

                "Proposal не прошёл Reviewer",

            review

        });

    }


    const quality =

        validateLearningQuality(
            proposal
        );


    if(
        quality.passed !== true
    ){

        return await buildRejectedResult({

            proposal,

            reason:

                quality.reason

                ||

                "Proposal не прошёл Quality Gate",

            review,

            quality

        });

    }


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );


    if(
        autonomy.action ===
        APPROVAL_ACTION.REINFORCE_ONLY
    ){

        return await persistSkillReinforcement({

            proposal,

            quality,

            autonomy

        });

    }


    if(
        autonomy.action !==
        APPROVAL_ACTION.AUTO_APPROVE
    ){

        return await buildKeepCandidateResult({

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

            "Jessica Learning Approval Processor error:",

            error

        );


        return await buildFailedResult({

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


    if(
        result?.success !== true
    ){

        return await buildFailedResult({

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


    const persistence =

        await markProposalApproved(
            proposal
        );


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


    return {

        ...result,

        success:
            true,

        learned:
            true,

        reinforced:
            false,

        versionCreated:
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
