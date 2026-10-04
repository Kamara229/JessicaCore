/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE CONSTANTS v2
 * =========================================================
 */


export const LEARNING_PROPOSALS_TABLE =
    "learning_proposals";


export const DEFAULT_PENDING_PROPOSALS_LIMIT =
    100;


export const PROPOSAL_STORAGE_STATUS = {

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


export const ALLOWED_PROPOSAL_STATUSES = [

    PROPOSAL_STORAGE_STATUS.PENDING_APPROVAL,

    PROPOSAL_STORAGE_STATUS.PROCESSING,

    PROPOSAL_STORAGE_STATUS.APPROVED,

    PROPOSAL_STORAGE_STATUS.KEEP_CANDIDATE,

    PROPOSAL_STORAGE_STATUS.REJECTED,

    PROPOSAL_STORAGE_STATUS.FAILED

];
