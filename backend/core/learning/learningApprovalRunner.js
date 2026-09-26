/*
 * =========================================================
 * JESSICA LEARNING APPROVAL RUNNER
 * =========================================================
 *
 * Финальный исполнитель обучения.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Approval Runner
 *        ↓
 * learningApproval
 *        ↓
 * Experience Skill
 *
 *
 * НЕ:
 *
 * - создаёт Proposal;
 * - анализирует обучение;
 * - меняет решение Reviewer.
 *
 * =========================================================
 */



import {
    approveAndSaveLearningProposal
} from "./learningApproval.js";





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





    try {


        const result =
            await approveAndSaveLearningProposal({

                proposal,

                skillId:
                    proposal.skillId || "",

                confidence:
                    proposal.confidence || 0

            });





        return result;



    } catch(error) {


        console.error(

            "Jessica Approval Runner error:",

            error

        );



        return {


            success:
                false,


            reason:
                error.message


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


        approved:

            results.filter(
                item =>
                    item.success === true
            ).length,


        failed:

            results.filter(
                item =>
                    item.success === false
            ).length,


        results


    };


}
