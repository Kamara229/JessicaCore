/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER v5
 * =========================================================
 *
 * Финальный координатор автономного обучения.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Reviewer
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
 * ожидание опыта
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - анализирует Execution Trace;
 * - вызывает AI.
 *
 * =========================================================
 */



import {
    reviewLearningProposal
} from "./learningReviewer.js";


import {
    evaluateLearningAutonomy
} from "./learningAutonomyPolicy.js";


import {
    approveAndSaveLearningProposal
} from "../../experience/learning/learningApproval.js";









/*
 * =========================================================
 * PROCESS SINGLE PROPOSAL
 * =========================================================
 */


async function processProposal(
    proposal
){

    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return {

            success:false,

            learned:false,

            reason:
                "Proposal отсутствует"

        };

    }








    /*
     * =====================================================
     * 1. REVIEW
     * =====================================================
     */


    const review =

        reviewLearningProposal(
            proposal
        );






    if(
        review.valid !== true
    ){

        return {


            success:true,


            learned:false,


            status:
                "REJECT",



            reason:

                review.reason,



            review


        };

    }









    /*
     * =====================================================
     * 2. AUTONOMY POLICY
     * =====================================================
     */


    const autonomy =

        evaluateLearningAutonomy(
            proposal
        );







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



            autonomy,


            review


        };

    }









    /*
     * =====================================================
     * 3. SAVE EXPERIENCE
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



            autonomy,


            review


        };





    }catch(error){


        console.error(

            "Jessica Learning Approval Runner error:",

            error

        );



        return {


            success:false,


            learned:false,


            status:
                "FAILED",



            reason:
                error.message,



            autonomy,


            review


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
){

    if(
        !Array.isArray(proposals)
    ){

        return {


            success:false,


            processed:0,


            learned:0,


            candidates:0,


            failed:0,


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

            )
            .length,



        candidates:

            results.filter(

                item =>
                    item.status ===
                    "KEEP_CANDIDATE"

            )
            .length,



        rejected:

            results.filter(

                item =>
                    item.status ===
                    "REJECT"

            )
            .length,



        failed:

            results.filter(

                item =>
                    item.success === false

            )
            .length,



        results


    };

}
