/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER v3
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
 * Learning Approval
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
 * - анализирует опыт;
 * - сохраняет память напрямую.
 *
 * =========================================================
 */



import {
    approveAndSaveLearningProposal
} from "../../experience/learning/learningApproval.js";


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


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return {

            success:false,

            reason:
                "Proposal отсутствует"

        };

    }







    /*
     * =====================================================
     * AUTONOMY DECISION
     * =====================================================
     */


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );







    /*
     * Сохраняем решение
     * только в текущий поток.
     *
     * Proposal в БД не изменяем.
     */


    if(
        autonomy.action !==
        "AUTO_APPROVE"
    ){

        return {


            success:true,


            learned:false,


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


                autonomy,


                confidence:

                    proposal.confidence || 0


            });






        return {


            ...result,


            learned:

                result.success === true,



            autonomy


        };




    } catch(error) {


        console.error(

            "Jessica Learning Approval Runner error:",

            error

        );



        return {


            success:false,


            learned:false,


            reason:

                error.message,



            autonomy


        };


    }


}









/*
 * =========================================================
 * RUN BATCH
 * =========================================================
 */


export async function runLearningApproval(
    proposals = []
) {


    if(
        !Array.isArray(proposals)
    ){

        return {


            success:false,


            processed:0,


            error:
                "Invalid proposals"


        };

    }







    const results = [];







    for(
        const proposal
        of proposals
    ){

        const result =

            await processProposal(
                proposal
            );


        results.push(
            result
        );

    }







    return {


        success:true,


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
