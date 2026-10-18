/*
 * =========================================================
 * JESSICA REINFORCEMENT PERSISTENCE v1
 * =========================================================
 *
 * READY Skill + unchanged knowledge:
 *
 * - append Evidence;
 * - do NOT create Skill version;
 * - mark Proposal APPROVED.
 * =========================================================
 */


import {
    appendExperienceEvidence
} from "../../../experience/storage/supabaseExperienceEvidenceWriter.js";


import {
    persistProposalStatus
} from "./approvalPersistence.js";


import {
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


import {
    promoteCandidateMemory
} from "./approvalCandidateMemory.js";


export async function persistSkillReinforcement({

    proposal,

    quality,

    autonomy

} = {}) {

    const skillId =

        String(

            proposal
                ?.targetSkill
                ?.id

            ||

            proposal
                ?.proposedExperience
                ?.id

            ||

            ""

        )
            .trim();


    const evidence =

        Array.isArray(
            proposal
                ?.proposedExperience
                ?.examples
        )

            ? proposal.proposedExperience.examples

            : [];


    const evidenceResult =

        await appendExperienceEvidence({

            skillId,

            evidence

        });


    if(
        evidenceResult?.success !== true
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

                evidenceResult?.error

                ||

                "Evidence reinforcement не сохранён",

            quality,

            autonomy,

            evidence:
                evidenceResult

        };

    }


    const persistence =

        await persistProposalStatus({

            proposal,

            status:
                APPROVAL_RESULT_STATUS.APPROVED

        });


    const candidateMemory =

        await promoteCandidateMemory({

            proposal,

            skillId

        });


    return {

        success:
            true,

        learned:
            false,

        reinforced:
            true,

        versionCreated:
            false,

        status:
            APPROVAL_RESULT_STATUS.APPROVED,

        reason:

            autonomy?.reason

            ||

            "READY Skill reinforced without new version",

        skillId,

        evidence:
            evidenceResult,

        quality,

        autonomy,

        persistence,

        candidateMemory

    };

}
