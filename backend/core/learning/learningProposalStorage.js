/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL STORAGE v3
 * =========================================================
 *
 * Persistent Storage для Learning Proposal.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Supabase
 *        ↓
 * learning_proposals
 *
 *
 * Lifecycle:
 *
 * PENDING_APPROVAL
 *        ↓
 *
 * ┌──────────────────┬──────────────────┬───────────────┐
 * ↓                  ↓                  ↓               ↓
 * APPROVED      KEEP_CANDIDATE      REJECTED         FAILED
 *
 *
 * Ответственность:
 *
 * - сохранить Learning Proposal;
 * - сохранить proposedExperience;
 * - сохранить Learning Analysis;
 * - сохранить Target Skill;
 * - сохранить provenance;
 * - изменить статус Proposal;
 * - установить approved_at / rejected_at;
 * - нормализовать Supabase Row
 *   в Runtime Proposal.
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - принимает Learning Decision;
 * - запускает Reviewer;
 * - запускает Quality Gate;
 * - запускает Autonomy;
 * - создаёт Experience Skill;
 * - сохраняет Experience Skill.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





const TABLE_NAME =
    "learning_proposals";





/*
 * =========================================================
 * STATUSES
 * =========================================================
 */


const ALLOWED_STATUSES = [

    "PENDING_APPROVAL",

    "APPROVED",

    "KEEP_CANDIDATE",

    "REJECTED",

    "FAILED"

];





/*
 * =========================================================
 * CLIENT
 * =========================================================
 */


function getClient()
{

    return getSupabaseClient();

}





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}





function normalizeObject(
    value
) {

    return isObject(
        value
    )

        ? value

        : {};

}





function normalizeStatus(
    value
) {

    const status =

        normalizeText(
            value
        )
        .toUpperCase();


    return ALLOWED_STATUSES.includes(
        status
    )

        ? status

        : null;

}





/*
 * =========================================================
 * ANALYSIS PAYLOAD
 * =========================================================
 *
 * В таблице уже существует JSONB analysis.
 *
 * Используем его также для сохранения:
 *
 * - provenance;
 * - traceId.
 *
 *
 * Благодаря этому не требуется
 * добавлять новые колонки Supabase
 * только ради служебного Learning Context.
 *
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
 * BUILD INSERT PAYLOAD
 * =========================================================
 */


function buildPayload(
    proposal
) {

    const status =

        normalizeStatus(
            proposal?.status
        )

        ||

        "PENDING_APPROVAL";


    return {


        id:

            proposal.id,



        status,



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



        /*
         * =================================================
         * FUTURE EXPERIENCE
         * =================================================
         */


        proposed_experience:

            normalizeObject(
                proposal.proposedExperience
            ),



        /*
         * =================================================
         * LEARNING CONTEXT
         * =================================================
         */


        analysis:

            buildAnalysisPayload(
                proposal
            ),



        /*
         * =================================================
         * TARGET EXPERIENCE
         * =================================================
         */


        target_skill:

            normalizeObject(
                proposal.targetSkill
            ),



        /*
         * =================================================
         * TIMESTAMPS
         * =================================================
         */


        created_at:

            proposal.createdAt

            ||

            new Date()
                .toISOString(),



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
 * NORMALIZE DATABASE PROPOSAL
 * =========================================================
 *
 * Supabase:
 *
 * queue_item_id
 * proposed_experience
 * target_skill
 * created_at
 *
 *
 * Runtime:
 *
 * queueItemId
 * proposedExperience
 * targetSkill
 * createdAt
 *
 *
 * DB-поля также сохраняем
 * для обратной совместимости.
 *
 * =========================================================
 */


function normalizeDatabaseProposal(
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

            normalizeStatus(
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



        proposedExperience:

            normalizeObject(

                row.proposed_experience

                ??

                row.proposedExperience

            ),



        analysis,



        targetSkill:

            normalizeObject(

                row.target_skill

                ??

                row.targetSkill

            ),



        provenance:

            normalizeObject(
                analysis.provenance
            ),



        createdAt:

            row.created_at

            ??

            row.createdAt

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

            normalizeObject(
                row.proposed_experience
            ),



        target_skill:

            normalizeObject(
                row.target_skill
            ),



        created_at:

            row.created_at

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





/*
 * =========================================================
 * VALIDATE PROPOSAL
 * =========================================================
 */


function validateProposalForStorage(
    proposal
) {

    if(
        !isObject(
            proposal
        )
    ){

        return {

            valid:
                false,

            error:
                "Invalid proposal"

        };

    }


    if(
        !normalizeText(
            proposal.id
        )
    ){

        return {

            valid:
                false,

            error:
                "Proposal ID отсутствует"

        };

    }


    if(
        !normalizeStatus(
            proposal.status
        )
    ){

        return {

            valid:
                false,

            error:
                "Некорректный Proposal status"

        };

    }


    if(
        !isObject(
            proposal.proposedExperience
        )
    ){

        return {

            valid:
                false,

            error:
                "proposedExperience отсутствует"

        };

    }


    return {

        valid:
            true

    };

}





/*
 * =========================================================
 * SAVE
 * =========================================================
 */


export async function saveLearningProposal(
    proposal
) {

    const validation =

        validateProposalForStorage(
            proposal
        );


    if(
        !validation.valid
    ){

        return {

            success:
                false,

            error:
                validation.error

        };

    }


    try {


        const payload =

            buildPayload(
                proposal
            );


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .insert(
                    payload
                )

                .select()

                .single();


        if(
            error
        ){

            console.error(

                "Jessica Learning Proposal insert error:",

                error

            );


            return {

                success:
                    false,

                error:
                    error.message

            };

        }


        return {

            success:
                true,

            proposal:

                normalizeDatabaseProposal(
                    data
                )

                ||

                data

        };


    }catch(error){


        console.error(

            "Jessica Learning Proposal storage error:",

            error

        );


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Proposal Storage error"

        };

    }

}





/*
 * =========================================================
 * BUILD STATUS UPDATE
 * =========================================================
 */


function buildStatusUpdate(
    status
) {

    const normalizedStatus =

        normalizeStatus(
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
        "APPROVED"
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
        "REJECTED"
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
     * =====================================================
     * NON-FINAL LEARNING STATES
     * =====================================================
     *
     * KEEP_CANDIDATE:
     *
     * Skill пока не одобрен
     * и не отклонён.
     *
     *
     * FAILED:
     *
     * техническая ошибка Pipeline,
     * а не semantic rejection.
     *
     *
     * Поэтому approved_at
     * и rejected_at не заполняются.
     *
     * =====================================================
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





/*
 * =========================================================
 * UPDATE STATUS
 * =========================================================
 */


export async function updateLearningProposalStatus(

    id,

    status

) {

    const proposalId =

        normalizeText(
            id
        );


    if(
        !proposalId
    ){

        return {

            success:
                false,

            error:
                "Proposal ID отсутствует"

        };

    }


    const updatePayload =

        buildStatusUpdate(
            status
        );


    if(
        !updatePayload
    ){

        return {

            success:
                false,

            error:

                `Unsupported Proposal status: ${normalizeText(status)}`

        };

    }


    try {


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .update(
                    updatePayload
                )

                .eq(
                    "id",
                    proposalId
                )

                .select()

                .single();


        if(
            error
        ){

            console.error(

                "Jessica Learning Proposal status update error:",

                error

            );


            return {

                success:
                    false,

                error:
                    error.message

            };

        }


        return {

            success:
                true,

            proposal:

                normalizeDatabaseProposal(
                    data
                )

                ||

                data

        };


    }catch(error){


        console.error(

            "Jessica Learning Proposal status storage error:",

            error

        );


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Proposal status update failed"

        };

    }

}





/*
 * =========================================================
 * STATUS SUPPORT
 * =========================================================
 */


export function isSupportedLearningProposalStatus(
    status
) {

    return Boolean(

        normalizeStatus(
            status
        )

    );

}
