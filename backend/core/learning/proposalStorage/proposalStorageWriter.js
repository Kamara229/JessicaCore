/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE WRITER
 * =========================================================
 *
 * Mutation-side Learning Proposal Storage.
 *
 *
 * Отвечает:
 *
 * - сохранить Proposal;
 * - обеспечить idempotency;
 * - изменить Proposal Status.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../../storage/supabaseClient.js";


import {
    LEARNING_PROPOSALS_TABLE
} from "./proposalStorageConstants.js";


import {
    isObject,
    normalizeText,
    normalizeProposalStatus
} from "./proposalStorageUtils.js";


import {
    buildProposalInsertPayload,
    normalizeDatabaseProposal
} from "./proposalStorageMapper.js";


import {
    buildProposalStatusUpdate
} from "./proposalStorageStatus.js";


import {
    findLearningProposalByQueueItemId
} from "./proposalStorageReader.js";





/*
 * =========================================================
 * VALIDATE
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
        !normalizeProposalStatus(
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


        /*
         * =================================================
         * IDEMPOTENCY PRE-CHECK
         * =================================================
         */


        if(
            proposal.queueItemId
        ){

            const existing =

                await findLearningProposalByQueueItemId(
                    proposal.queueItemId
                );


            if(
                !existing.success
            ){

                return {

                    success:
                        false,

                    error:

                        existing.error

                        ||

                        "Не удалось проверить существующий Proposal"

                };

            }


            if(
                existing.found
                &&
                existing.proposal
            ){

                return {

                    success:
                        true,

                    existing:
                        true,

                    proposal:
                        existing.proposal

                };

            }

        }


        /*
         * =================================================
         * INSERT
         * =================================================
         */


        const payload =

            buildProposalInsertPayload(
                proposal
            );


        const {
            data,
            error
        } =

            await getSupabaseClient()

                .from(
                    LEARNING_PROPOSALS_TABLE
                )

                .insert(
                    payload
                )

                .select()

                .single();


        /*
         * =================================================
         * UNIQUE CONFLICT RECOVERY
         * =================================================
         *
         * После добавления UNIQUE INDEX
         * два параллельных процесса
         * могут одновременно пройти
         * предварительный lookup.
         *
         * База разрешит только одному INSERT.
         *
         * Второй получит PostgreSQL 23505.
         *
         * Для Learning это не ошибка:
         * просто возвращаем существующий Proposal.
         *
         * =================================================
         */


        if(
            error?.code === "23505"
            &&
            proposal.queueItemId
        ){

            const existing =

                await findLearningProposalByQueueItemId(
                    proposal.queueItemId
                );


            if(
                existing.success
                &&
                existing.found
            ){

                return {

                    success:
                        true,

                    existing:
                        true,

                    conflictRecovered:
                        true,

                    proposal:
                        existing.proposal

                };

            }

        }


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
                    error.message,

                code:
                    error.code || null

            };

        }


        return {

            success:
                true,

            existing:
                false,

            conflictRecovered:
                false,

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

        buildProposalStatusUpdate(
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

            await getSupabaseClient()

                .from(
                    LEARNING_PROPOSALS_TABLE
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

        normalizeProposalStatus(
            status
        )

    );

}
