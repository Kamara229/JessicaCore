/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL STORAGE v2
 * =========================================================
 *
 * Хранилище Learning Proposal.
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
 * Отвечает:
 *
 * - сохранение Proposal;
 * - сохранение полного контекста обучения.
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - принимает решение;
 * - создаёт Skill.
 *
 * =========================================================
 */



import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





const TABLE_NAME =
    "learning_proposals";









function getClient()
{

    return getSupabaseClient();

}









/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function safeNumber(
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






function safeObject(
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
 * BUILD PAYLOAD
 * =========================================================
 */


function buildPayload(
    proposal
){

    return {


        id:

            proposal.id,



        status:

            proposal.status,



        source:

            proposal.source ||
            "learning_queue",



        queue_item_id:

            proposal.queueItemId || null,



        action:

            proposal.action || null,



        confidence:

            safeNumber(
                proposal.confidence
            ),






        /*
         * Полный Experience Candidate
         */


        proposed_experience:

            safeObject(
                proposal.proposedExperience
            ),






        /*
         * Анализ пригодности
         */


        analysis:

            safeObject(
                proposal.analysis
            ),






        /*
         * Целевой Skill
         */


        target_skill:

            safeObject(
                proposal.targetSkill
            ),






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
 * SAVE
 * =========================================================
 */


export async function saveLearningProposal(
    proposal
){

    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return {


            success:false,


            error:
                "Invalid proposal"


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

            return {


                success:false,


                error:
                    error.message


            };

        }







        return {


            success:true,


            proposal:
                data


        };







    } catch(error){



        return {


            success:false,


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


export async function updateLearningProposalStatus(
    id,
    status
){

    if(
        !id ||
        !status
    ){

        return {


            success:false,


            error:
                "Missing id or status"


        };

    }





    try{


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .update({

                    status

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


            proposal:
                data


        };



    }catch(error){


        return {


            success:false,


            error:
                error.message


        };


    }


}
