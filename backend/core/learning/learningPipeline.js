/*
 * =========================================================
 * JESSICA LEARNING PIPELINE
 * =========================================================
 *
 * Полный цикл обработки обучения Jessica.
 *
 *
 * Flow:
 *
 * Learning Queue
 *       ↓
 * Learning Worker
 *       ↓
 * Learning Proposal
 *       ↓
 * Learning Reviewer
 *       ↓
 * Approval
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - изменяет Experience;
 * - пишет напрямую в память.
 *
 * =========================================================
 */


import {
    runLearningWorker
} from "./learningWorker.js";


import {
    reviewLearningProposal
} from "./learningReviewer.js";





/*
 * =========================================================
 * PROCESS PROPOSALS
 * =========================================================
 */


function reviewWorkerResults(
    workerResult
) {


    const results =
        Array.isArray(
            workerResult?.results
        )
            ? workerResult.results
            : [];



    return results.map(

        item => {


            if (
                !item.success ||
                !item.proposal
            ) {


                return {

                    success:
                        false,

                    proposal:
                        null,

                    review:
                        null,

                    reason:
                        item.reason ||
                        "Proposal не создан"

                };

            }





            const review =
                reviewLearningProposal(
                    item.proposal
                );





            return {

                success:
                    review.approved === true,


                proposal:
                    item.proposal,


                review


            };


        }

    );

}





/*
 * =========================================================
 * RUN LEARNING PIPELINE
 * =========================================================
 */


export async function runLearningPipeline()
{


    /*
     * =====================================================
     * 1. WORKER
     * =====================================================
     */


    const workerResult =
        await runLearningWorker();





    if (
        !workerResult?.success
    ) {


        return {


            success:
                false,


            stage:
                "worker",


            error:
                workerResult?.error ||
                "Learning Worker failed"


        };

    }





    /*
     * =====================================================
     * 2. REVIEW
     * =====================================================
     */


    const reviewed =
        reviewWorkerResults(
            workerResult
        );





    return {


        success:
            true,


        stage:
            "review",


        processed:
            workerResult.processed || 0,


        approvedReady:

            reviewed.filter(
                item =>
                    item.review?.status ===
                    "APPROVE_READY"
            ).length,


        clarificationNeeded:

            reviewed.filter(
                item =>
                    item.review?.status ===
                    "NEEDS_CLARIFICATION"
            ).length,


        rejected:

            reviewed.filter(
                item =>
                    item.review?.status ===
                    "REJECT"
            ).length,


        results:
            reviewed


    };


}
