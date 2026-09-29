/*
 * =========================================================
 * JESSICA LEARNING QUEUE STORAGE v2
 * =========================================================
 *
 * Persistent storage для Learning Queue.
 *
 *
 * Flow:
 *
 * Learning Event
 *        ↓
 * Learning Queue
 *        ↓
 * Learning Worker
 *        ↓
 * Learning Proposal
 *
 *
 * Ответственность:
 *
 * - сохранить Learning Event;
 * - получить ожидающие события;
 * - изменить статус обработки.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - создаёт Proposal;
 * - принимает решение обучения;
 * - создаёт Skill.
 *
 * =========================================================
 */



import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





const TABLE_NAME =
    "learning_queue";









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


function normalizeText(
    value
){

    return String(
        value || ""
    )
    .trim();

}






function normalizeNumber(
    value
){

    const number =
        Number(value);



    return Number.isFinite(number)
        ?
        number
        :
        0;

}






function normalizeObject(
    value
){

    if(
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ){

        return {};

    }


    return value;

}









/*
 * =========================================================
 * VALIDATION
 * =========================================================
 */


function isValidQueueItem(
    item
){

    if(
        !item ||
        typeof item !== "object"
    ){

        return false;

    }



    const allowedActions = [

        "NEW_SKILL",

        "SKILL_IMPROVEMENT"

    ];



    return allowedActions.includes(
        item.action
    );

}









/*
 * =========================================================
 * SAVE QUEUE ITEM
 * =========================================================
 */


export async function saveLearningQueueItem(
    item
){

    if(
        !isValidQueueItem(
            item
        )
    ){

        return {


            success:false,


            error:
                "Invalid learning queue item"


        };

    }






    try {



        const payload = {



            id:

                item.id || null,



            skill_id:

                normalizeText(
                    item.skillId
                )
                ||
                null,



            action:

                item.action,



            confidence:

                normalizeNumber(
                    item.confidence
                ),



            status:

                item.status ||
                "PENDING",



            event_json:

                normalizeObject(
                    item.event
                ),



            created_at:

                item.createdAt

                ||

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







        if(
            error
        ){

            console.error(

                "Jessica Learning Queue insert error:",

                error

            );



            return {


                success:false,


                error:
                    error.message


            };

        }







        return {


            success:true,


            id:
                data?.id || null,


            item:
                data


        };







    }catch(error){


        console.error(

            "Jessica Learning Queue storage error:",

            error

        );



        return {


            success:false,


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

                        ascending:true

                    }

                );







        if(
            error
        ){

            return {


                success:false,


                items:[],


                error:
                    error.message


            };

        }







        return {


            success:true,


            items:

                Array.isArray(data)

                ?

                data

                :

                []


        };






    }catch(error){


        return {


            success:false,


            items:[],


            error:
                error.message ||
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

)
{


    if(
        !id
    ){

        return {


            success:false,


            error:
                "Queue item id required"


        };

    }






    if(
        !status
    ){

        return {


            success:false,


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


                    processed_at:

                        new Date()
                            .toISOString()


                })

                .eq(
                    "id",
                    id
                )

                .select()

                .single();







        if(
            error
        ){

            return {


                success:false,


                error:
                    error.message


            };

        }








        return {


            success:true,


            item:
                data


        };







    }catch(error){


        return {


            success:false,


            error:
                error.message ||
                "Storage error"


        };


    }


}
