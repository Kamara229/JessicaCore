/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE STATUS
 * =========================================================
 *
 * Формирует DB update
 * для изменения Proposal Status.
 *
 * =========================================================
 */


import {
    normalizeProposalStatus
} from "./proposalStorageUtils.js";


import {
    PROPOSAL_STORAGE_STATUS
} from "./proposalStorageConstants.js";


export function buildProposalStatusUpdate(
    status
) {

    const normalizedStatus =

        normalizeProposalStatus(
            status
        );


    if(
        !normalizedStatus
    ){

        return null;

    }


    const now =

        new Date()
            .toISOString();


    /*
     * =====================================================
     * APPROVED
     * =====================================================
     */


    if(
        normalizedStatus ===
        PROPOSAL_STORAGE_STATUS.APPROVED
    ){

        return {

            status:
                normalizedStatus,

            approved_at:
                now,

            rejected_at:
                null

        };

    }


    /*
     * =====================================================
     * REJECTED
     * =====================================================
     */


    if(
        normalizedStatus ===
        PROPOSAL_STORAGE_STATUS.REJECTED
    ){

        return {

            status:
                normalizedStatus,

            approved_at:
                null,

            rejected_at:
                now

        };

    }


    /*
     * KEEP_CANDIDATE
     * FAILED
     * PENDING_APPROVAL
     *
     * не являются approval/rejection.
     */


    return {

        status:
            normalizedStatus,

        approved_at:
            null,

        rejected_at:
            null

    };

}
