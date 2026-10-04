/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE STATUS v2
 * =========================================================
 *
 * Формирует DB payload
 * изменения Proposal Status.
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
     * PROCESSING
     * =====================================================
     */


    if(
        normalizedStatus ===
        PROPOSAL_STORAGE_STATUS.PROCESSING
    ){

        return {

            status:
                normalizedStatus,

            processing_at:
                now,

            approved_at:
                null,

            rejected_at:
                null

        };

    }


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

            processing_at:
                null,

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

            processing_at:
                null,

            approved_at:
                null,

            rejected_at:
                now

        };

    }


    /*
     * =====================================================
     * KEEP_CANDIDATE
     * FAILED
     * PENDING_APPROVAL
     * =====================================================
     */


    return {

        status:
            normalizedStatus,

        processing_at:
            null,

        approved_at:
            null,

        rejected_at:
            null

    };

}
