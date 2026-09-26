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
 * Supabase
 *
 *
 * Ответственность:
 *
 * - сохранить Learning Event;
 * - получить ожидающие события;
 * - обновить статус.
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
    supabase
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
 * SAVE QUEUE ITEM
 * =========================================================
 */


export async function saveLearningQueueItem(
    item
) {


    if (
        !item ||
        typeof item !== "object"
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
            await supabase

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
                "Learning Queue save error:",
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
            "Learning Queue storage exception:",
            error
        );


        return {


            success:
                false,


            error:
                error.message ||
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
            await supabase

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
                error.message


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





    try {


        const {
            data,
            error
        } =
            await supabase

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
                error.message


        };

    }


}
