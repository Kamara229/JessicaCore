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
import {reconcileCandidatePromotion} from "./approvalCandidateReconciliation.js";


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


    // The atomic save may have committed while its ACK was lost.
    // Do not mark Proposal FAILED: stale PROCESSING recovery can replay it.
    if(result?.success!==true && result?.uncertain===true){
        return {...result,success:false,learned:false,deferred:true,
            status:"PROCESSING",proposalStateUpdated:false,
            candidateMemoryStateUpdated:false};
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


    const publication=classifyApprovalPublication(result);

    // The Experience version is durable, but a crash here must leave the
    // Proposal PROCESSING until its linked Candidate is PROMOTED.
    const candidateMemory=await reconcileCandidatePromotion({
        proposal,
        skillId:result.skillId||result?.experience?.id||null,
        promote
    });
    if(candidateMemory.success!==true){
        return {
            ...result,success:false,learned:false,reinforced:false,
            versionCreated:publication.versionCreated,
            publicationRecovered:publication.publicationRecovered,
            deferred:true,status:APPROVAL_RESULT_STATUS.PROCESSING,
            proposalStateUpdated:false,
            proposalStateError:"Candidate PROMOTED must finish first",
            candidateMemoryStateUpdated:false,
            candidateMemoryStateError:candidateMemory.error,
            review,quality,autonomy,candidateMemory
        };
    }

    // Only finalize the Proposal after Candidate promotion has succeeded.
    // A failed or lost status ACK is recoverable; promotion is idempotent.
    let persistence;
    try{
        persistence=await markApproved(proposal);
    }catch(error){
        persistence={success:false,
            error:error?.message||"Proposal approval persistence failed"};
    }
    const approved=persistence?.success===true;
    return {
        ...result,
        success:approved,learned:approved,reinforced:false,
        versionCreated:publication.versionCreated,
        publicationRecovered:publication.publicationRecovered,
        deferred:!approved,
        status:approved?APPROVAL_RESULT_STATUS.APPROVED:
            APPROVAL_RESULT_STATUS.PROCESSING,
        proposalStateUpdated:approved,
        proposalStateError:approved?null:
            (persistence?.error||"Proposal approval unconfirmed"),
        candidateMemoryStateUpdated:true,
        candidateMemoryStateError:null,
        review,quality,autonomy,persistence,candidateMemory
    };

}
