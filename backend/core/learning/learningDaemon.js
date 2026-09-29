/*
 * =========================================================
 * JESSICA LEARNING DAEMON v4
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
 * Experience Memory
 *
 *
 * НЕ:
 *
 * - выполняет пользовательские задачи;
 * - вызывает AI;
 * - анализирует опыт;
 * - создаёт Skill напрямую.
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


function emptyResult(
    reason
){

    return {


        success:true,


        reason:


            reason || "nothing-to-learn",



        pipeline:null,


        approval:null


    };

}









/*
 * =========================================================
 * SINGLE CYCLE
 * =========================================================
 */


export async function runLearningDaemonCycle()
{


    if(
        running
    ){

        return {


            success:false,


            reason:
                "Learning cycle already running"


        };

    }






    running =
        true;







    try {



        /*
         * =============================================
         * 1. PIPELINE
         * =============================================
         */


        const pipelineResult =

            await runLearningPipeline();






        if(
            !pipelineResult?.success
        ){

            return {


                success:false,


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








        if(
            proposals.length === 0
        ){

            return emptyResult(
                "no-proposals"
            );

        }









        /*
         * =============================================
         * 2. APPROVAL RUNNER
         * =============================================
         */


        let approvalResult;



        try {


            approvalResult =

                await runLearningApproval(
                    proposals
                );



        }catch(error){


            return {


                success:false,


                stage:
                    "approval",



                pipeline:

                    pipelineResult,



                error:
                    error.message


            };

        }









        return {


            success:true,



            pipeline:
                pipelineResult,



            proposals:

                proposals.length,



            learned:

                approvalResult?.learned || 0,



            candidates:

                approvalResult?.candidates || 0,



            failed:

                approvalResult?.failed || 0,



            approval:

                approvalResult


        };






    }catch(error){


        console.error(

            "Jessica Learning Daemon error:",

            error

        );



        return {


            success:false,


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


            started:false,


            reason:
                "Daemon already started"


        };

    }





    daemonTimer =

        setInterval(

            async () => {


                await runLearningDaemonCycle();



            },

            interval

        );





    console.log(

        "Jessica Learning Daemon started"

    );





    return {


        started:true,


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


            stopped:false


        };

    }




    clearInterval(
        daemonTimer
    );



    daemonTimer =
        null;



    return {


        stopped:true


    };

}
