/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE RECOVERY v1
 * =========================================================
 *
 * Восстанавливает Learning Proposal,
 * зависшие в PROCESSING после:
 *
 * - Render restart;
 * - process crash;
 * - deploy;
 * - unexpected shutdown.
 *
 *
 * Flow:
 *
 * PROCESSING
 *      ↓
 * stale timeout
 *      ↓
 * PENDING_APPROVAL
 *      ↓
 * следующий Atomic Claim
 *
 *
 * ВАЖНО:
 *
 * Recovery безопасен только потому,
 * что Experience Persistence теперь
 * идемпотентен по proposalId.
 *
 *
 * НЕ:
 *
 * - выполняет Approval;
 * - создаёт Experience;
 * - claim'ит Proposal;
 * - принимает Learning Decision.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../../storage/supabaseClient.js";


const RECOVERY_RPC =
    "recover_stale_learning_proposals";


const DEFAULT_STALE_MINUTES =
    30;


const DEFAULT_RECOVERY_LIMIT =
    100;


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizePositiveInteger(
    value,
    fallback
) {

    const number =
        Math.floor(
            Number(value)
        );


    if(
        !Number.isInteger(number)
        ||
        number < 1
    ){

        return fallback;

    }


    return number;

}


/*
 * =========================================================
 * ENV CONFIG
 * =========================================================
 */


function resolveStaleMinutes()
{

    const configured =

        normalizePositiveInteger(

            process.env
                .LEARNING_PROCESSING_STALE_MINUTES,

            DEFAULT_STALE_MINUTES

        );


    /*
     * Не разрешаем случайно поставить
     * слишком маленькое значение.
     *
     * Иначе живой Approval может быть
     * ошибочно возвращён в очередь.
     */


    return Math.max(
        configured,
        5
    );

}


function resolveRecoveryLimit()
{

    return Math.min(

        normalizePositiveInteger(

            process.env
                .LEARNING_PROCESSING_RECOVERY_LIMIT,

            DEFAULT_RECOVERY_LIMIT

        ),

        500

    );

}


/*
 * =========================================================
 * RECOVER
 * =========================================================
 */


export async function recoverStaleLearningProposals({

    staleMinutes =
        resolveStaleMinutes(),

    limit =
        resolveRecoveryLimit()

} = {}) {


    const safeStaleMinutes =

        Math.max(

            normalizePositiveInteger(

                staleMinutes,

                DEFAULT_STALE_MINUTES

            ),

            5

        );


    const safeLimit =

        Math.min(

            normalizePositiveInteger(

                limit,

                DEFAULT_RECOVERY_LIMIT

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

                    RECOVERY_RPC,

                    {

                        p_stale_minutes:
                            safeStaleMinutes,

                        p_limit:
                            safeLimit

                    }

                );


        if(
            error
        ){

            console.error(

                "Jessica Learning Proposal recovery error:",

                error

            );


            return {

                success:
                    false,

                recovered:
                    0,

                error:
                    error.message

            };

        }


        const result =

            Array.isArray(data)

                ? data[0]

                : data;


        const recovered =

            Math.max(

                Number(
                    result?.recovered || 0
                ),

                0

            );


        return {

            success:
                true,

            recovered,

            staleMinutes:
                safeStaleMinutes,

            limit:
                safeLimit

        };


    }catch(error){


        console.error(

            "Jessica Learning Proposal recovery failure:",

            error

        );


        return {

            success:
                false,

            recovered:
                0,

            staleMinutes:
                safeStaleMinutes,

            error:

                error?.message

                ||

                "Proposal recovery failed"

        };

    }

}
