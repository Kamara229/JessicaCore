/*
 * =========================================================
 * JESSICA EXECUTION TRACE v4
 * =========================================================
 *
 * История выполнения одного Execution Run.
 *
 *
 * Используется:
 *
 * - debugging;
 * - analytics;
 * - Learning;
 * - Experience Analyzer.
 *
 *
 * Хранит:
 *
 * - события выполнения;
 * - шаги;
 * - ошибки;
 * - применённый Experience;
 * - итог.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает Planner;
 * - изменяет Skills;
 * - обучает систему.
 *
 * =========================================================
 */



import {
    randomUUID
} from "node:crypto";









/*
 * =========================================================
 * CREATE TRACE
 * =========================================================
 */


export function createExecutionTrace(

    task

) {


    return {


        id:

            randomUUID(),



        task:

            String(
                task || ""
            )
            .trim(),



        startedAt:

            new Date()
                .toISOString(),



        finishedAt:

            null,



        status:

            "RUNNING",






        /*
         * =================================================
         * EVENTS
         * =================================================
         */


        events:

            [],






        /*
         * =================================================
         * EXECUTION STEPS
         * =================================================
         */


        steps:

            [],







        /*
         * =================================================
         * FAILURES
         * =================================================
         */


        failures:

            [],






        /*
         * =================================================
         * EXPERIENCE USAGE
         * =================================================
         */


        experienceUsage:

        {


            used:

                false,



            source:

                null,



            skills:

                []

        },








        /*
         * =================================================
         * RESULT
         * =================================================
         */


        result:

            null,



        learningReady:

            false


    };


}









/*
 * =========================================================
 * ADD EVENT
 * =========================================================
 */


export function addTraceEvent(

    trace,

    type,

    payload = {}

) {


    if (
        !trace
    ) {

        return trace;

    }




    trace.events.push({

        id:

            randomUUID(),



        type,



        payload,



        timestamp:

            new Date()
                .toISOString()

    });



    return trace;


}









/*
 * =========================================================
 * ADD STEP
 * =========================================================
 */


export function addTraceStep(

    trace,

    step

) {


    if (
        !trace ||
        !step
    ) {

        return trace;

    }



    trace.steps.push({

        ...normalizeStep(
            step
        ),


        timestamp:

            new Date()
                .toISOString()

    });



    return trace;


}









/*
 * =========================================================
 * UPDATE FROM EXECUTION RESULT
 * =========================================================
 */


export function updateTraceFromResult(

    trace,

    result

) {


    if (
        !trace ||
        !result
    ) {

        return trace;

    }






    addTraceEvent(

        trace,

        result.success === true

            ? "EXECUTION_COMPLETED"

            : "EXECUTION_FAILED",

        result

    );








    if (
        result.failure
    ) {


        trace.failures.push({

            ...result.failure,


            timestamp:

                new Date()
                    .toISOString()

        });


    }








    collectExperience(

        trace,

        result

    );






    trace.result =
        result;



    return trace;


}









/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


function collectExperience(

    trace,

    result

) {


    const experience =

        result
            ?.executionMeta
            ?.experience;



    if (
        !experience ||
        experience.used !== true
    ) {

        return;

    }






    trace.experienceUsage.used =
        true;




    trace.experienceUsage.source =

        experience.source ||
        trace.experienceUsage.source;






    if (
        Array.isArray(
            experience.skills
        )
    ) {



        for (
            const skill
            of experience.skills
        ) {


            const id =

                skill?.id ||
                skill;



            const exists =

                trace.experienceUsage.skills
                    .some(

                        item =>

                            (
                                item?.id ||
                                item
                            )
                            ===
                            id

                    );



            if (
                !exists
            ) {

                trace.experienceUsage.skills.push(
                    skill
                );

            }


        }


    }



}









/*
 * =========================================================
 * UPDATE SUMMARY
 * =========================================================
 */


export function updateTraceFromSummary(

    trace,

    summary

) {


    if (
        !trace ||
        !summary
    ) {

        return trace;

    }





    addTraceEvent(

        trace,

        "EXECUTION_SUMMARY",

        summary

    );




    if (
        Array.isArray(
            summary.results
        )
    ) {


        for (
            const item
            of summary.results
        ) {


            updateTraceFromResult(

                trace,

                item

            );


        }


    }





    return trace;


}









/*
 * =========================================================
 * NORMALIZE STEP
 * =========================================================
 */


function normalizeStep(

    step

) {


    return {


        id:

            step.id ||
            null,



        tool:

            step.tool ||
            null,



        success:

            step.success === true,



        result:

            step.result ||
            null



    };


}









/*
 * =========================================================
 * FINISH TRACE
 * =========================================================
 */


export function finishExecutionTrace(

    trace

) {


    if (
        !trace
    ) {

        return trace;

    }



    trace.finishedAt =

        new Date()
            .toISOString();






    trace.status =

        trace.result?.success === true

            ? "COMPLETED"

            : "FAILED";






    trace.learningReady = true;



    return trace;


}









/*
 * =========================================================
 * BUILD LEARNING PAYLOAD
 * =========================================================
 */


export function buildLearningPayload(

    trace

) {


    if (
        !trace
    ) {

        return null;

    }




    return {


        executionId:

            trace.id,



        task:

            trace.task,



        success:

            trace.status === "COMPLETED",



        experience:

            trace.experienceUsage,



        steps:

            trace.steps,



        failures:

            trace.failures,



        events:

            trace.events,



        result:

            trace.result


    };


}
