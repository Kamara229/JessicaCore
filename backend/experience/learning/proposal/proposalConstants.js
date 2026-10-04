/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL CONSTANTS v2
 * =========================================================
 */


export const LEARNING_PROPOSAL_STATUS = {

    PENDING_APPROVAL:
        "PENDING_APPROVAL",

    PROCESSING:
        "PROCESSING",

    APPROVED:
        "APPROVED",

    KEEP_CANDIDATE:
        "KEEP_CANDIDATE",

    REJECTED:
        "REJECTED",

    FAILED:
        "FAILED"

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
