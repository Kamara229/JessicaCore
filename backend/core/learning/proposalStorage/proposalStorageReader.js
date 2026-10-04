/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE READER
 * =========================================================
 *
 * Read-side Learning Proposal Storage.
 *
 *
 * Отвечает:
 *
 * - найти Proposal по Queue Item;
 * - получить PENDING_APPROVAL;
 * - вернуть Runtime Proposal.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../../storage/supabaseClient.js";


import {
    LEARNING_PROPOSALS_TABLE,
    DEFAULT_PENDING_PROPOSALS_LIMIT
} from "./proposalStorageConstants.js";


import {
    normalizeText,
    normalizePositiveInteger
} from "./proposalStorageUtils.js";


import {
    normalizeDatabaseProposal
} from "./proposalStorageMapper.js";


/*
 * =========================================================
 * FIND BY QUEUE ITEM
 * =========================================================
 */


export async function findLearningProposalByQueueItemId(
    queueItemId
) {

    const id =

        normalizeText(
            queueItemId
        );


    if(
        !id
    ){

        return {

            success:
                true,

            found:
                false,

            proposal:
                null

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

                .select("*")

                .eq(
                    "queue_item_id",
                    id
                )

                .order(
                    "created_at",
                    {
                        ascending:
                            true
                    }
                )

                .limit(1)

                .maybeSingle();


        if(
            error
        ){

            return {

                success:
                    false,

                found:
                    false,

                proposal:
                    null,

                error:
                    error.message

            };

        }


        if(
            !data
        ){

            return {

                success:
                    true,

                found:
                    false,

                proposal:
                    null

            };

        }


        return {

            success:
                true,

            found:
                true,

            proposal:

                normalizeDatabaseProposal(
                    data
                )

        };


    }catch(error){


        return {

            success:
                false,

            found:
                false,

            proposal:
                null,

            error:

                error?.message

                ||

                "Proposal lookup failed"

        };

    }

}


/*
 * =========================================================
 * GET PENDING
 * =========================================================
 */


export async function getPendingLearningProposals({

    limit =
        DEFAULT_PENDING_PROPOSALS_LIMIT

} = {}) {


    const safeLimit =

        normalizePositiveInteger(

            limit,

            DEFAULT_PENDING_PROPOSALS_LIMIT

        );


    try {


        const {
            data,
            error
        } =

            await getSupabaseClient()

                .from(
                    LEARNING_PROPOSALS_TABLE
                )

                .select("*")

                .eq(
                    "status",
                    "PENDING_APPROVAL"
                )

                .order(
                    "created_at",
                    {
                        ascending:
                            true
                    }
                )

                .limit(
                    safeLimit
                );


        if(
            error
        ){

            return {

                success:
                    false,

                proposals:
                    [],

                count:
                    0,

                error:
                    error.message

            };

        }


        const rows =

            Array.isArray(
                data
            )

                ? data

                : [];


        const proposals =

            rows

                .map(
                    normalizeDatabaseProposal
                )

                .filter(
                    Boolean
                );


        return {

            success:
                true,

            proposals,

            count:
                proposals.length

        };


    }catch(error){


        return {

            success:
                false,

            proposals:
                [],

            count:
                0,

            error:

                error?.message

                ||

                "Pending Proposal read failed"

        };

    }

}
