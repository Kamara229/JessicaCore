/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE MAPPER v2
 * =========================================================
 *
 * Runtime Proposal
 *        ↓
 * Supabase Payload
 *
 * Supabase Row
 *        ↓
 * Runtime Proposal
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText,
    normalizeNumber,
    normalizeObject,
    normalizeProposalStatus
} from "./proposalStorageUtils.js";


/*
 * =========================================================
 * ANALYSIS PAYLOAD
 * =========================================================
 */


function buildAnalysisPayload(
    proposal
) {

    const analysis =

        normalizeObject(
            proposal?.analysis
        );


    const provenance =

        normalizeObject(
            proposal?.provenance
        );


    return {

        ...analysis,

        provenance,

        traceId:

            proposal?.traceId

            ||

            provenance?.traceId

            ||

            null

    };

}


/*
 * =========================================================
 * INSERT PAYLOAD
 * =========================================================
 */


export function buildProposalInsertPayload(
    proposal
) {

    return {

        id:

            proposal.id,


        status:

            normalizeProposalStatus(
                proposal.status
            )

            ||

            "PENDING_APPROVAL",


        source:

            normalizeText(
                proposal.source
            )

            ||

            "learning_queue",


        queue_item_id:

            proposal.queueItemId

            ||

            null,


        action:

            normalizeText(
                proposal.action
            )

            ||

            null,


        confidence:

            normalizeNumber(
                proposal.confidence
            ),


        proposed_experience:

            normalizeObject(
                proposal.proposedExperience
            ),


        analysis:

            buildAnalysisPayload(
                proposal
            ),


        target_skill:

            normalizeObject(
                proposal.targetSkill
            ),


        created_at:

            proposal.createdAt

            ||

            new Date()
                .toISOString(),


        processing_at:

            proposal.processingAt

            ||

            null,


        approved_at:

            proposal.approvedAt

            ||

            null,


        rejected_at:

            proposal.rejectedAt

            ||

            null

    };

}


/*
 * =========================================================
 * DATABASE → RUNTIME
 * =========================================================
 */


export function normalizeDatabaseProposal(
    row
) {

    if(
        !isObject(
            row
        )
    ){

        return null;

    }


    const analysis =

        normalizeObject(
            row.analysis
        );


    const provenance =

        normalizeObject(
            analysis.provenance
        );


    const candidateMemory =

        isObject(
            analysis.candidateMemory
        )

            ? analysis.candidateMemory

            : (
                provenance.candidateMemoryId

                    ? {

                        id:

                            provenance.candidateMemoryId,


                        key:

                            provenance.candidateMemoryKey

                            ||

                            null

                    }

                    : null
            );


    const proposedExperience =

        normalizeObject(

            row.proposed_experience

            ??

            row.proposedExperience

        );


    const targetSkill =

        normalizeObject(

            row.target_skill

            ??

            row.targetSkill

        );


    return {

        /*
         * =================================================
         * RUNTIME CONTRACT
         * =================================================
         */


        id:

            row.id

            ||

            null,


        status:

            normalizeProposalStatus(
                row.status
            )

            ||

            normalizeText(
                row.status
            )

            ||

            null,


        source:

            normalizeText(
                row.source
            )

            ||

            null,


        queueItemId:

            row.queue_item_id

            ??

            row.queueItemId

            ??

            null,


        traceId:

            analysis.traceId

            ??

            provenance.traceId

            ??

            null,


        action:

            normalizeText(
                row.action
            )

            ||

            null,


        confidence:

            normalizeNumber(
                row.confidence
            ),


        proposedExperience,


        analysis,


        targetSkill,


        provenance,


        candidateMemory,


        createdAt:

            row.created_at

            ??

            row.createdAt

            ??

            null,


        processingAt:

            row.processing_at

            ??

            row.processingAt

            ??

            null,


        approvedAt:

            row.approved_at

            ??

            row.approvedAt

            ??

            null,


        rejectedAt:

            row.rejected_at

            ??

            row.rejectedAt

            ??

            null,


        /*
         * =================================================
         * DB COMPATIBILITY
         * =================================================
         */


        queue_item_id:

            row.queue_item_id

            ??

            null,


        proposed_experience:

            proposedExperience,


        target_skill:

            targetSkill,


        created_at:

            row.created_at

            ??

            null,


        processing_at:

            row.processing_at

            ??

            null,


        approved_at:

            row.approved_at

            ??

            null,


        rejected_at:

            row.rejected_at

            ??

            null

    };

}
