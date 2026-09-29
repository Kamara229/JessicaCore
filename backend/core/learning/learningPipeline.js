/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v4
 * =========================================================
 *
 * Координатор первого этапа обучения.
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
 * Approval Runner
 *
 *
 * Ответственность:
 *
 * - запустить Worker;
 * - получить Proposal;
 * - передать дальше.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - принимает решение;
 * - создаёт Skill;
 * - сохраняет Experience.
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
){

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

                item?.success === true
                &&
                item?.proposal

        )

        .map(

            item =>
                item.proposal

        );


}









/*
 * =========================================================
 * STATS
 * =========================================================
 */


function buildStats(
    workerResult
){

    const results =

        Array.isArray(
            workerResult?.results
        )
        ?
        workerResult.results
        :
        [];



    return {


        processed:

            results.length,



        successful:

            results.filter(

                item =>
                    item.success === true

            )
            .length,



        failed:

            results.filter(

                item =>
                    item.success !== true

            )
            .length



    };


}









/*
 * =========================================================
 * RUN PIPELINE
 * =========================================================
 */


export async function runLearningPipeline()
{


    let workerResult;



    try{


        workerResult =

            await runLearningWorker();



    }catch(error){


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
        !workerResult ||
        workerResult.success !== true
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









    const proposals =

        extractProposals(
            workerResult
        );






    return {


        success:true,


        stage:
            "proposal",



        stats:

            buildStats(
                workerResult
            ),




        worker:

            workerResult,




        created:

            proposals.length,




        proposals



    };


}
