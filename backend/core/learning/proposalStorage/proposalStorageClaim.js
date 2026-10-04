/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE CLAIM v1
 * =========================================================
 *
 * Атомарно забирает Learning Proposal
 * для обработки текущим Daemon Cycle.
 *
 *
 * Database:
 *
 * PENDING_APPROVAL
 *        ↓
 * FOR UPDATE SKIP LOCKED
 *        ↓
 * PROCESSING
 *        ↓
 * RETURNING Proposal
 *
 *
 * Благодаря этому два процесса
 * не смогут получить один Proposal.
 *
 *
 * НЕ:
 *
 * - принимает Learning Decision;
 * - сохраняет Experience;
 * - возвращает stale PROCESSING;
 * - выполняет Approval.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../../storage/supabaseClient.js";


import {
    DEFAULT_PENDING_PROPOSALS_LIMIT
} from "./proposalStorageConstants.js";


import {
    normalizePositiveInteger
} from "./proposalStorageUtils.js";


import {
    normalizeDatabaseProposal
} from "./proposalStorageMapper.js";


const CLAIM_RPC =
    "claim_learning_proposals";


/*
 * =========================================================
 * CLAIM
 * =========================================================
 */


export async function claimPendingLearningProposals({

    limit =
        DEFAULT_PENDING_PROPOSALS_LIMIT

} = {}) {


    const safeLimit =

        Math.min(

            normalizePositiveInteger(

                limit,

                DEFAULT_PENDING_PROPOSALS_LIMIT

            ),

            500

        );


    try {


        const {
            data,
            error
        } =

            await getSupabaseClient()

                .rpc(

                    CLAIM_RPC,

                    {

                        p_limit:
                            safeLimit

                    }

                );


        if(
            error
        ){

            console.error(

                "Jessica Learning Proposal claim error:",

                error

            );


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


        console.error(

            "Jessica Learning Proposal claim failure:",

            error

        );


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

                "Proposal claim failed"

        };

    }

}
