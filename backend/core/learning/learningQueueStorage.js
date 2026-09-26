/*
 * =========================================================
 * JESSICA LEARNING QUEUE STORAGE
 * =========================================================
 *
 * Persistent storage для Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Queue Item
 *        ↓
 * Learning Queue Storage
 *        ↓
 * Supabase Client
 *        ↓
 * PostgreSQL
 *
 *
 * Ответственность:
 *
 * - сохранить Learning Event;
 * - получить ожидающие события;
 * - обновить статус обучения.
 *
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - создаёт Skill;
 * - принимает решение Approval.
 *
 * =========================================================
 */



import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





/*
 * =========================================================
 * TABLE
 * =========================================================
 */


const TABLE_NAME =
    "learning_queue";





/*
 * =========================================================
 * CLIENT
 * =========================================================
 *
 * Клиент создаётся через общий Supabase слой.
 *
 * =========================================================
 */


function getClient()
{

    return getSupabaseClient();

}





/*
 * =========================================================
 * VALIDATION
 * =========================================================
 */


function isValidQueueItem(
    item
) {


    return (

        item &&

        typeof item === "object"

    );


}





/*
 * =========================================================
 * SAVE QUEUE ITEM
 * =========================================================
 */


export async function saveLearningQueueItem(
    item
) {


    if (
        !isValidQueueItem(
            item
        )
    ) {


        return {

            success:
                false,

            error:
                "Invalid queue item"

        };

    }





    try {


        const payload = {


            id:
                item.id || null,


            skill_id:
                item.skillId || null,


            action:
                item.action || "IGNORE",


            confidence:
                Number(
                    item.confidence || 0
                ),


            status:
                item.status || "PENDING",


            event_json:
                item.event || {},


            created_at:
                item.createdAt ||
                new Date()
                    .toISOString()


        };





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





        if (
            error
        ) {


            console.error(
                "Jessica Learning Queue insert error:",
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


            id:
                data?.id || null,


            item:
                data


        };



    } catch(error) {


        console.error(
            "Jessica Learning Queue storage exception:",
            error
        );


        return {


            success:
                false,


            error:
                error?.message ||
                "Storage error"


        };

    }


}





/*
 * =========================================================
 * GET PENDING ITEMS
 * =========================================================
 */


export async function getPendingLearningItems()
{


    try {


        const {
            data,
            error
        } =
            await getClient()

                .from(
                    TABLE_NAME
                )

                .select("*")

                .eq(
                    "status",
                    "PENDING"
                )

                .order(
                    "created_at",
                    {

                        ascending:
                            true

                    }
                );





        if (
            error
        ) {


            return {


                success:
                    false,


                items:
                    [],


                error:
                    error.message


            };

        }





        return {


            success:
                true,


            items:
                data || []


        };



    } catch(error) {


        return {


            success:
                false,


            items:
                [],


            error:
                error?.message ||
                "Storage error"


        };

    }


}





/*
 * =========================================================
 * UPDATE STATUS
 * =========================================================
 */


export async function updateLearningQueueItemStatus(

    id,

    status

) {


    if (
        !id
    ) {


        return {


            success:
                false,


            error:
                "Queue item id required"


        };

    }





    if (
        !status
    ) {


        return {


            success:
                false,


            error:
                "Queue status required"


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

                .update({

                    status,

                    reviewed_at:
                        new Date()
                            .toISOString()

                })

                .eq(
                    "id",
                    id
                )

                .select()

                .single();





        if (
            error
        ) {


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


            item:
                data


        };



    } catch(error) {


        return {


            success:
                false,


            error:
                error?.message ||
                "Storage error"


        };

    }


}
