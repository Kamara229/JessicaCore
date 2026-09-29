/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v3
 * =========================================================
 *
 * Первый этап Learning Pipeline.
 *
 *
 * Flow:
 *
 * Learning Queue
 *       ↓
 * Learning Worker
 *       ↓
 * Learning Proposal
 *
 *
 * Ответственность:
 *
 * - запустить Learning Worker;
 * - получить созданные Proposal;
 * - передать их дальше.
 *
 *
 * НЕ:
 *
 * - принимает решение обучения;
 * - вызывает Reviewer;
 * - создаёт Skill;
 * - сохраняет Experience.
 *
 *
 * Решение:
 *
 * Autonomy Policy
 *
 * =========================================================
 */



import {
    runLearningWorker
} from "./learningWorker.js";








/*
 * =========================================================
 * EXTRACT PROPOSALS
 * =========================================================
 */


function extractProposals(
    workerResult
) {


    if(
        !Array.isArray(
            workerResult?.results
        )
    ){

        return [];

    }



    return workerResult.results

        .filter(

            item =>

                item.success === true
                &&
                item.proposal

        )

        .map(

            item =>

                item.proposal

        );


}









/*
 * =========================================================
 * RUN PIPELINE
 * =========================================================
 */


export async function runLearningPipeline()
{


    /*
     * =====================================================
     * WORKER
     * =====================================================
     */


    let workerResult;



    try {


        workerResult =

            await runLearningWorker();



    } catch(error) {


        return {


            success:false,


            stage:
                "worker",


            error:
                error.message,


            proposals:[]

        };


    }







    if(
        !workerResult?.success
    ){


        return {


            success:false,


            stage:
                "worker",


            error:

                workerResult?.error

                ||

                "Learning Worker failed",



            proposals:[]

        };


    }








    /*
     * =====================================================
     * PROPOSALS
     * =====================================================
     */


    const proposals =

        extractProposals(
            workerResult
        );









    return {


        success:true,


        stage:
            "proposal",



        processed:

            workerResult.processed || 0,



        created:

            proposals.length,



        proposals



    };


}
