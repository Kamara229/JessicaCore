/*
 * =========================================================
 * JESSICA APPROVAL PERSISTENCE
 * =========================================================
 *
 * Сохраняет финальное состояние
 * Learning Proposal.
 *
 *
 * НЕ:
 *
 * - принимает решение;
 * - сохраняет Experience Skill;
 * - анализирует Proposal.
 *
 * =========================================================
 */


import {
    updateLearningProposalStatus
} from "../learningProposalStorage.js";


export async function persistProposalStatus({

    proposal,

    status

} = {}) {


    if(
        !proposal?.id
    ){

        return {

            success:
                false,

            error:
                "Proposal ID отсутствует"

        };

    }


    if(
        !status
    ){

        return {

            success:
                false,

            error:
                "Proposal status отсутствует"

        };

    }


    try {


        return await updateLearningProposalStatus(

            proposal.id,

            status

        );


    }catch(error){


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Proposal status persistence failed"

        };

    }

}
