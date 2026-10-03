/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL CONSTANTS
 * =========================================================
 *
 * Общие Actions и Status
 * Learning Proposal.
 *
 * =========================================================
 */


export const LEARNING_PROPOSAL_STATUS = {

    PENDING_APPROVAL:
        "PENDING_APPROVAL",

    APPROVED:
        "APPROVED",

    REJECTED:
        "REJECTED"

};


export const LEARNING_PROPOSAL_ACTION = {

    NEW_SKILL:
        "NEW_SKILL",

    SKILL_IMPROVEMENT:
        "SKILL_IMPROVEMENT"

};


export const VALID_PROPOSAL_ACTIONS = [

    LEARNING_PROPOSAL_ACTION.NEW_SKILL,

    LEARNING_PROPOSAL_ACTION.SKILL_IMPROVEMENT

];
