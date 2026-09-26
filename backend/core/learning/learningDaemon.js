/*
 * =========================================================
 * JESSICA LEARNING DAEMON
 * =========================================================
 *
 * Фоновый процесс обучения Jessica.
 *
 *
 * Flow:
 *
 * Timer
 *   ↓
 * Learning Pipeline
 *   ↓
 * Reviewer
 *   ↓
 * Approval Runner
 *   ↓
 * Experience Memory
 *
 *
 * НЕ:
 *
 * - выполняет пользовательские задачи;
 * - влияет на ответы напрямую;
 * - создаёт Skill без Approval.
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


    if (
        running
    ) {


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
         * =============================================
         * 1. CREATE AND REVIEW PROPOSALS
         * =============================================
         */


        const pipelineResult =
            await runLearningPipeline();





        const approvalReady =
            pipelineResult
                ?.results
                ?.filter(

                    item =>

                        item.review?.status ===
                        "APPROVE_READY"

                )
                .map(

                    item =>
                        item.proposal

                )
                || [];





        /*
         * =============================================
         * 2. SAVE SKILLS
         * =============================================
         */


        let approvalResult =
            null;



        if (
            approvalReady.length > 0
        ) {


            approvalResult =
                await runLearningApproval(
                    approvalReady
                );

        }





        return {


            success:
                true,


            pipeline:
                pipelineResult,


            approval:
                approvalResult


        };



    } catch(error) {


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
) {


    if (
        daemonTimer
    ) {


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


                runLearningDaemonCycle();



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
 * STOP
 * =========================================================
 */


export function stopLearningDaemon()
{


    if (
        !daemonTimer
    ) {


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
