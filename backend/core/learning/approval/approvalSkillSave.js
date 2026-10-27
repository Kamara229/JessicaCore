/*
 * =========================================================
 * JESSICA APPROVAL SKILL SAVE
 * =========================================================
 *
 * Сохраняет новую Experience Version
 * после успешных Reviewer / Quality / Autonomy.
 * =========================================================
 */

import {
    approveAndSaveLearningProposal
} from "../../../experience/learning/learningApproval.js";


import {
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


import {
    promoteCandidateMemory
} from "./approvalCandidateMemory.js";


import {
    buildFailedResult,
    markProposalApproved
} from "./approvalResultBuilders.js";
import {classifyApprovalPublication} from "./approvalPublicationOutcome.js";


export async function saveApprovedSkillVersion({

    proposal,

    review,

    quality,

    autonomy

}, {
    approve=approveAndSaveLearningProposal,
    markApproved=markProposalApproved,
    promote=promoteCandidateMemory
} = {}) {

    let result;


    try {

        result =

            await approve({

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

            "Jessica Learning Approval Save error:",

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

        await markApproved(
            proposal
        );


    const candidateMemory =

        await promote({

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


    const publication=classifyApprovalPublication(result);

    return {

        ...result,

        success:
            true,

        learned:
            true,

        reinforced:
            false,

        versionCreated:
            publication.versionCreated,

        publicationRecovered:
            publication.publicationRecovered,

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
