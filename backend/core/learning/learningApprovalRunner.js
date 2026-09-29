/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER v2
 * =========================================================
 *
 * Автономный исполнитель обучения.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Autonomy Policy
 *        ↓
 *
 * AUTO_APPROVE
 *        ↓
 * learningApproval
 *        ↓
 * Experience Skill
 *
 *
 * KEEP_CANDIDATE
 *        ↓
 * ожидание накопления опыта
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - анализирует обучение;
 * - вызывает AI;
 * - хранит память.
 *
 * =========================================================
 */



import {
    approveAndSaveLearningProposal
} from "./learningApproval.js";


import {
    evaluateLearningAutonomy
} from "./learningAutonomyPolicy.js";







/*
 * =========================================================
 * PROCESS SINGLE PROPOSAL
 * =========================================================
 */


async function processProposal(
    proposal
) {


    if (
        !proposal
    ) {


        return {

            success:
                false,

            reason:
                "Proposal отсутствует"

        };

    }







    /*
     * =====================================================
     * AUTONOMY CHECK
     * =====================================================
     */


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );





    if (
        autonomy.action !==
        "AUTO_APPROVE"
    ) {


        return {


            success:
                true,


            learned:
                false,


            status:
                "KEEP_CANDIDATE",


            reason:
                autonomy.reason,


            autonomy


        };

    }









    /*
     * =====================================================
     * SAVE EXPERIENCE SKILL
     * =====================================================
     */


    try {


        const result =

            await approveAndSaveLearningProposal({

                proposal,


                skillId:

                    proposal.skillId || "",



                confidence:

                    proposal.confidence || 0


            });






        return {


            ...result,


            autonomy

        };




    } catch(error) {


        console.error(

            "Jessica Approval Runner error:",

            error

        );



        return {


            success:
                false,


            reason:
                error.message,


            autonomy

        };


    }


}









/*
 * =========================================================
 * RUN APPROVAL BATCH
 * =========================================================
 */


export async function runLearningApproval(
    proposals = []
) {


    if (
        !Array.isArray(
            proposals
        )
    ) {


        return {


            success:
                false,


            processed:
                0,


            error:
                "Invalid proposals"

        };

    }






    const results =
        [];





    for (
        const proposal
        of proposals
    ) {


        const result =

            await processProposal(
                proposal
            );


        results.push(
            result
        );


    }








    return {


        success:
            true,


        processed:

            proposals.length,



        learned:

            results.filter(

                item =>
                    item.learned === true

            ).length,



        candidates:

            results.filter(

                item =>
                    item.status ===
                    "KEEP_CANDIDATE"

            ).length,



        failed:

            results.filter(

                item =>
                    item.success === false

            ).length,



        results


    };


}
