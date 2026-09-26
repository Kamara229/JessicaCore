/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL STORAGE
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
 * НЕ:
 *
 * - создаёт Proposal;
 * - принимает Approval;
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
 * SAVE PROPOSAL
 * =========================================================
 */


export async function saveLearningProposal(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {

        return {

            success:
                false,

            error:
                "Invalid proposal"

        };

    }



    try {


        const payload = {


            id:
                proposal.id,


            status:
                proposal.status,


            source:
                proposal.source || "learning_queue",


            queue_item_id:
                proposal.queueItemId || null,


            action:
                proposal.action || null,


            confidence:
                proposal.confidence || 0,


            proposed_experience:
                proposal.proposedExperience || {},


            created_at:
                proposal.createdAt ||
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
