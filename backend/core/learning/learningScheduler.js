/*
 * =========================================================
 * JESSICA LEARNING SCHEDULER
 * =========================================================
 *
 * Автоматический запуск Learning Pipeline.
 *
 *
 * Flow:
 *
 * Timer
 *    ↓
 * Learning Pipeline
 *    ↓
 * Queue Processing
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - создаёт Proposal;
 * - создаёт Skill;
 * - работает с Supabase напрямую.
 *
 * =========================================================
 */



import {
    runLearningPipeline
} from "./learningPipeline.js";





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const DEFAULT_INTERVAL =
    15 * 60 * 1000;





let schedulerTimer =
    null;





/*
 * =========================================================
 * RUN ONCE
 * =========================================================
 */


export async function runLearningCycle()
{


    try {


        const result =
            await runLearningPipeline();



        console.log(

            "Jessica Learning Cycle:",

            JSON.stringify(
                result
            )

        );



        return result;



    } catch(error) {


        console.error(

            "Jessica Learning Scheduler error:",

            error

        );



        return {

            success:
                false,

            error:
                error.message

        };

    }


}





/*
 * =========================================================
 * START SCHEDULER
 * =========================================================
 */


export function startLearningScheduler(

    interval =
        DEFAULT_INTERVAL

) {


    if (
        schedulerTimer
    ) {

        return {

            started:
                false,

            reason:
                "Scheduler already running"

        };

    }





    console.log(
        "Jessica Learning Scheduler started"
    );





    schedulerTimer =

        setInterval(

            () => {


                runLearningCycle();


            },

            interval

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


export function stopLearningScheduler()
{


    if (
        !schedulerTimer
    ) {

        return {

            stopped:
                false

        };

    }





    clearInterval(
        schedulerTimer
    );



    schedulerTimer =
        null;





    return {


        stopped:
            true

    };


}
