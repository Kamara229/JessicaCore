/*
 * =========================================================
 * JESSICA LEARNING DAEMON v3
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
 * Autonomy Policy
 *   ↓
 * Approval Runner
 *   ↓
 * Experience Memory
 *
 *
 * НЕ:
 *
 * - выполняет пользовательские задачи;
 * - вызывает AI;
 * - принимает решение вручную.
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
         * 1. CREATE PROPOSALS
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








        /*
         * =============================================
         * 2. GET PROPOSALS
         * =============================================
         */


        const proposals =

            Array.isArray(
                pipelineResult.proposals
            )

            ?

            pipelineResult.proposals

            :

            [];









        /*
         * =============================================
         * 3. AUTONOMOUS APPROVAL
         * =============================================
         */


        let approvalResult =
            null;





        if(
            proposals.length > 0
        ){

            approvalResult =

                await runLearningApproval(
                    proposals
                );

        }








        return {


            success:true,


            pipeline:

                pipelineResult,



            proposals:

                proposals.length,



            approval:

                approvalResult


        };





    } catch(error) {


        console.error(

            "Jessica Learning Daemon error:",

            error

        );



        return {


            success:false,


            error:
                error.message


        };



    } finally {


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
)
{


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

            () => {


                runLearningDaemonCycle();



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
 * STOP
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
