/*
 * =========================================================
 * JESSICA LEARNING DAEMON v5
 * =========================================================
 *
 * Фоновый процесс автономного обучения Jessica.
 *
 *
 * Flow:
 *
 * Timer
 *   ↓
 * Learning Pipeline
 *   ↓
 * Learning Proposal
 *   ↓
 * Approval Runner
 *   ↓
 * Autonomy Policy
 *   ↓
 * Learning Approval
 *   ↓
 * Experience Memory
 *
 *
 * Ответственность:
 *
 * - запускать цикл обучения;
 * - координировать Pipeline;
 * - передавать Proposal в Approval Runner;
 * - возвращать статистику обучения.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - создаёт Proposal;
 * - создаёт Skill;
 * - работает с БД.
 *
 * =========================================================
 */



import {
    runLearningPipeline
} from "./learningPipeline.js";


import {
    runLearningApproval
} from "./learningApprovalRunner.js";









/*
 * =========================================================
 * STATE
 * =========================================================
 */


let daemonTimer =
    null;



let running =
    false;









/*
 * =========================================================
 * EMPTY RESULT
 * =========================================================
 */


function createEmptyCycleResult(
    reason,
    pipeline = null
){

    return {


        success:
            true,



        reason:

            reason ||

            "nothing-to-learn",



        pipeline,



        proposals:
            0,



        learned:
            0,



        candidates:
            0,



        failed:
            0,



        approval:
            null


    };


}









/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
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









/*
 * =========================================================
 * SINGLE LEARNING CYCLE
 * =========================================================
 */


export async function runLearningDaemonCycle()
{


    if(
        running
    ){

        return {


            success:
                false,


            reason:

                "Learning cycle already running"


        };

    }







    running =
        true;






    try {



        /*
         * =================================================
         * 1. LEARNING PIPELINE
         * =================================================
         */


        const pipelineResult =

            await runLearningPipeline();







        if(
            !pipelineResult?.success
        ){

            return {


                success:
                    false,


                stage:
                    "pipeline",



                error:

                    pipelineResult?.error

                    ||

                    "Learning Pipeline failed"



            };

        }








        const proposals =

            Array.isArray(
                pipelineResult.proposals
            )

            ?

            pipelineResult.proposals

            :

            [];








        /*
         * =================================================
         * NOTHING TO APPROVE
         * =================================================
         */


        if(
            proposals.length === 0
        ){

            return createEmptyCycleResult(

                "no-proposals",

                pipelineResult

            );

        }









        /*
         * =================================================
         * 2. APPROVAL RUNNER
         * =================================================
         */


        let approvalResult;



        try {


            approvalResult =

                await runLearningApproval(
                    proposals
                );



        }catch(error){


            return {


                success:
                    false,


                stage:
                    "approval",



                pipeline:
                    pipelineResult,



                proposals:
                    proposals.length,



                error:
                    error.message



            };


        }









        /*
         * =================================================
         * FINAL RESULT
         * =================================================
         */


        return {


            success:
                true,



            pipeline:
                pipelineResult,



            proposals:
                proposals.length,



            learned:

                normalizeNumber(

                    approvalResult?.learned

                ),



            candidates:

                normalizeNumber(

                    approvalResult?.candidates

                ),



            failed:

                normalizeNumber(

                    approvalResult?.failed

                ),



            approval:
                approvalResult



        };







    }catch(error){



        console.error(

            "Jessica Learning Daemon error:",

            error

        );



        return {


            success:
                false,


            error:
                error.message



        };





    }finally{


        running =
            false;


    }


}









/*
 * =========================================================
 * START DAEMON
 * =========================================================
 */


export function startLearningDaemon(
    interval = 15 * 60 * 1000
){

    if(
        daemonTimer
    ){

        return {


            started:
                false,


            reason:

                "Daemon already started"



        };

    }







    daemonTimer =

        setInterval(

            () => {


                runLearningDaemonCycle()

                    .catch(error => {


                        console.error(

                            "Learning Daemon cycle error:",

                            error

                        );


                    });



            },

            interval

        );







    console.log(

        "Jessica Learning Daemon started"

    );







    return {


        started:
            true,


        interval



    };


}









/*
 * =========================================================
 * STOP DAEMON
 * =========================================================
 */


export function stopLearningDaemon()
{


    if(
        !daemonTimer
    ){

        return {


            stopped:
                false



        };

    }







    clearInterval(
        daemonTimer
    );



    daemonTimer =
        null;






    return {


        stopped:
            true



    };


}
