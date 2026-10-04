/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL VALIDATION v2
 * =========================================================
 *
 * Structural validation
 * Learning Proposal.
 *
 *
 * ACTIVE proposal states:
 *
 * PENDING_APPROVAL
 * PROCESSING
 *
 *
 * PROCESSING означает:
 *
 * Proposal атомарно захвачен
 * текущим Approval Cycle.
 *
 * =========================================================
 */


import {
    LEARNING_PROPOSAL_ACTION,
    LEARNING_PROPOSAL_STATUS
} from "./proposalConstants.js";


import {
    isObject,
    normalizeText
} from "./proposalUtils.js";





/*
 * =========================================================
 * CANDIDATE CONTRACT
 * =========================================================
 */


export function validateCandidateForProposal({

    candidate,

    targetSkill,

    action

}) {


    if(
        !isObject(
            candidate
        )
    ){

        throw new Error(
            "Learning Proposal: Skill Candidate отсутствует"
        );

    }


    if(
        !normalizeText(
            candidate.name
        )
    ){

        throw new Error(
            "Learning Proposal: Candidate name отсутствует"
        );

    }


    if(
        !Array.isArray(
            candidate.workflow
        )
        ||
        candidate.workflow.length === 0
    ){

        throw new Error(
            "Learning Proposal: Candidate workflow отсутствует"
        );

    }


    if(
        !Array.isArray(
            candidate.examples
        )
        ||
        candidate.examples.length === 0
    ){

        throw new Error(
            "Learning Proposal: Candidate examples отсутствуют"
        );

    }


    if(
        !targetSkill?.id
    ){

        throw new Error(
            "Learning Proposal: Target Skill ID отсутствует"
        );

    }


    if(
        action ===
        LEARNING_PROPOSAL_ACTION.SKILL_IMPROVEMENT

        &&

        targetSkill.exists !== true
    ){

        throw new Error(
            "Learning Proposal: Existing Skill не определён для Improvement"
        );

    }


    return true;

}





/*
 * =========================================================
 * ACTIVE PROPOSAL STATE
 * =========================================================
 */


export function validatePendingProposal(
    proposal
) {


    if(
        !isObject(
            proposal
        )
    ){

        throw new Error(
            "Learning Proposal отсутствует"
        );

    }


    const activeStates = [

        LEARNING_PROPOSAL_STATUS.PENDING_APPROVAL,

        LEARNING_PROPOSAL_STATUS.PROCESSING

    ];


    if(
        !activeStates.includes(
            proposal.status
        )
    ){

        throw new Error(
            "Learning Proposal уже обработан"
        );

    }


    return true;

}
