import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL v6
 * =========================================================
 *
 * Центральный координатор
 * Learning Proposal.
 *
 *
 * Flow:
 *
 * Queue Item
 *      ↓
 * Resolver
 *      ↓
 * Candidate Validation
 *      ↓
 * Proposed Experience Builder
 *      ↓
 * Proposal Analysis
 *      ↓
 * Learning Proposal
 *
 *
 * Proposal != Experience Skill
 *
 *
 * Ответственность:
 *
 * - координировать создание Proposal;
 * - создать Proposal ID;
 * - сохранить Target Skill;
 * - сохранить Candidate Memory reference;
 * - сохранить provenance;
 * - менять in-memory статус Proposal.
 *
 *
 * НЕ:
 *
 * - рассчитывает Candidate similarity;
 * - объединяет Candidate Memory;
 * - рассчитывает Learning Metrics;
 * - сохраняет Proposal;
 * - сохраняет Skill;
 * - принимает AUTO_APPROVE;
 * - работает с Supabase.
 *
 * =========================================================
 */


import {
    LEARNING_PROPOSAL_STATUS,
    LEARNING_PROPOSAL_ACTION
} from "./proposal/proposalConstants.js";


import {
    isObject,
    normalizeText
} from "./proposal/proposalUtils.js";


import {
    extractProposalEvent,
    resolveProposalAction,
    resolveProposalCandidate,
    resolveProposalConfidence,
    resolveProposalTargetSkill
} from "./proposal/proposalResolver.js";


import {
    buildProposedExperience
} from "./proposal/proposalExperienceBuilder.js";


import {
    buildProposalAnalysis
} from "./proposal/proposalAnalysis.js";


import {
    validateCandidateForProposal,
    validatePendingProposal
} from "./proposal/proposalValidation.js";





/*
 * =========================================================
 * PUBLIC CONSTANTS
 * =========================================================
 */


export {

    LEARNING_PROPOSAL_STATUS,

    LEARNING_PROPOSAL_ACTION

};





/*
 * =========================================================
 * TRACE ID
 * =========================================================
 */


function resolveTraceId(
    event
) {

    return (

        event?.traceId

        ||

        event
            ?.payload
            ?.discovery
            ?.traceId

        ||

        event
            ?.analysis
            ?.metadata
            ?.traceId

        ||

        null

    );

}





/*
 * =========================================================
 * CANDIDATE MEMORY REF
 * =========================================================
 */


function resolveCandidateMemory(
    event
) {

    const memory =

        event
            ?.payload
            ?.candidateMemory;


    if(
        !isObject(
            memory
        )
    ){

        return null;

    }


    if(
        !memory.id
    ){

        return null;

    }


    return {

        id:

            memory.id,


        key:

            normalizeText(
                memory.key
            )

            ||

            null,


        status:

            normalizeText(
                memory.status
            )

            ||

            null,


        matchType:

            normalizeText(
                memory.matchType
            )

            ||

            null,


        similarity:

            Number.isFinite(
                Number(
                    memory.similarity
                )
            )

                ? Number(
                    memory.similarity
                )

                : null,


        occurrences:

            Number.isFinite(
                Number(
                    memory.occurrences
                )
            )

                ? Number(
                    memory.occurrences
                )

                : null

    };

}





/*
 * =========================================================
 * CREATE FROM QUEUE
 * =========================================================
 */


export function createLearningProposalFromQueue(
    queueItem
) {


    /*
     * =====================================================
     * 1. QUEUE ITEM
     * =====================================================
     */


    if(
        !isObject(
            queueItem
        )
    ){

        throw new Error(
            "Learning Proposal: Queue Item отсутствует"
        );

    }



    /*
     * =====================================================
     * 2. EVENT
     * =====================================================
     */


    const event =

        extractProposalEvent(
            queueItem
        );


    if(
        Object.keys(
            event
        ).length === 0
    ){

        throw new Error(
            "Learning Proposal: Learning Event отсутствует"
        );

    }



    /*
     * =====================================================
     * 3. ACTION
     * =====================================================
     */


    const action =

        resolveProposalAction(

            queueItem,

            event

        );



    /*
     * =====================================================
     * 4. CANDIDATE
     * =====================================================
     */


    const candidate =

        resolveProposalCandidate(
            event
        );


    if(
        !candidate
    ){

        throw new Error(
            "Learning Proposal: Skill Candidate отсутствует в event.payload"
        );

    }



    /*
     * =====================================================
     * 5. TARGET SKILL
     * =====================================================
     */


    const targetSkill =

        resolveProposalTargetSkill({

            queueItem,

            event,

            candidate,

            action

        });



    /*
     * =====================================================
     * 6. VALIDATE
     * =====================================================
     */


    validateCandidateForProposal({

        candidate,

        targetSkill,

        action

    });



    /*
     * =====================================================
     * 7. CONFIDENCE
     * =====================================================
     */


    const confidence =

        resolveProposalConfidence({

            queueItem,

            event,

            candidate

        });



    /*
     * =====================================================
     * 8. PROPOSED EXPERIENCE
     * =====================================================
     */


    const proposedExperience =

        buildProposedExperience({

            candidate,

            targetSkill,

            confidence

        });



    /*
     * =====================================================
     * 9. ANALYSIS
     * =====================================================
     */


    const analysis =

        buildProposalAnalysis({

            event,

            candidate,

            confidence,

            action

        });



    /*
     * =====================================================
     * 10. REFERENCES
     * =====================================================
     */


    const traceId =

        resolveTraceId(
            event
        );


    const candidateMemory =

        resolveCandidateMemory(
            event
        );



    /*
     * =====================================================
     * 11. PROPOSAL
     * =====================================================
     */


    return {

        id:

            randomUUID(),


        status:

            LEARNING_PROPOSAL_STATUS
                .PENDING_APPROVAL,


        source:

            normalizeText(
                event?.payload?.source
            )

            ||

            "learning_queue",


        queueItemId:

            queueItem.id

            ||

            null,


        traceId,


        action,


        confidence,


        analysis,


        targetSkill: {

            id:
                targetSkill.id,

            version:
                targetSkill.version,

            exists:
                targetSkill.exists

        },


        /*
         * Persistent Candidate lineage.
         */


        candidateMemory,


        proposedExperience,


        provenance: {

            queueItemId:

                queueItem.id

                ||

                null,


            traceId,


            eventId:

                event.id

                ||

                null,


            source:

                normalizeText(
                    event?.payload?.source
                )

                ||

                "learning_queue",


            candidateType:

                normalizeText(
                    candidate.candidateType
                )

                ||

                action,


            dynamicPattern:

                event
                    ?.payload
                    ?.source ===
                    "ai-pattern-discovery",


            candidateMemoryId:

                candidateMemory?.id

                ||

                null,


            candidateMemoryKey:

                candidateMemory?.key

                ||

                null

        },


        createdAt:

            new Date()
                .toISOString(),


        approvedAt:
            null,


        rejectedAt:
            null

    };

}





/*
 * =========================================================
 * INTERNAL PROPOSAL
 * =========================================================
 */


export function createLearningProposal({

    task,

    proposedExperience = null

} = {}) {


    const cleanTask =

        normalizeText(
            task
        );


    if(
        !cleanTask
    ){

        throw new Error(
            "Learning Proposal: задача не указана"
        );

    }


    return {

        id:

            randomUUID(),


        status:

            LEARNING_PROPOSAL_STATUS
                .PENDING_APPROVAL,


        source:
            "internal",


        queueItemId:
            null,


        traceId:
            null,


        action:

            LEARNING_PROPOSAL_ACTION
                .NEW_SKILL,


        task:

            cleanTask,


        confidence:
            0,


        analysis: {

            reusable:
                false,

            reason:
                "internal-proposal",

            confidence:
                0,

            source:
                "internal",

            candidateMemory:
                null

        },


        targetSkill: {

            id:

                proposedExperience?.id

                ||

                proposedExperience?.skillId

                ||

                null,

            version:
                null,

            exists:
                false

        },


        candidateMemory:
            null,


        proposedExperience:

            isObject(
                proposedExperience
            )

                ? proposedExperience

                : null,


        provenance: {

            source:
                "internal",

            dynamicPattern:
                false,

            candidateMemoryId:
                null,

            candidateMemoryKey:
                null

        },


        createdAt:

            new Date()
                .toISOString(),


        approvedAt:
            null,


        rejectedAt:
            null

    };

}





/*
 * =========================================================
 * APPROVE
 * =========================================================
 */


export function approveLearningProposal(
    proposal
) {

    validatePendingProposal(
        proposal
    );


    return {

        ...proposal,

        status:

            LEARNING_PROPOSAL_STATUS
                .APPROVED,

        approvedAt:

            new Date()
                .toISOString(),

        rejectedAt:
            null

    };

}





/*
 * =========================================================
 * REJECT
 * =========================================================
 */


export function rejectLearningProposal(
    proposal
) {

    validatePendingProposal(
        proposal
    );


    return {

        ...proposal,

        status:

            LEARNING_PROPOSAL_STATUS
                .REJECTED,

        approvedAt:
            null,

        rejectedAt:

            new Date()
                .toISOString()

    };

}
