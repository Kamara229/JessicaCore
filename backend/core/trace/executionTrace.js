/*
 * =========================================================
 * JESSICA EXECUTION TRACE v5
 * =========================================================
 *
 * История одного Execution Run.
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
 * - события;
 * - попытки;
 * - шаги;
 * - ошибки;
 * - replans;
 * - Experience usage;
 * - итоговый результат.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает Planner;
 * - меняет Skills;
 * - принимает решения.
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
         * Все события цикла
         */


        events:

            [],








        /*
         * Попытки выполнения
         */


        attempts:

            [],









        /*
         * Шаги выполнения
         */


        steps:

            [],









        /*
         * Ошибки
         */


        failures:

            [],









        /*
         * Replan history
         */


        replans:

            [],









        /*
         * Experience
         */


        experienceUsage:

        {


            used:

                false,



            skills:

                [],



            source:

                null



        },









        /*
         * Финальный результат
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
 * ADD ATTEMPT
 * =========================================================
 */


export function addTraceAttempt(

    trace,

    attempt,

    data = {}

) {


    if (
        !trace
    ) {

        return trace;

    }




    trace.attempts.push({

        attempt,

        ...data,


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

        id:

            step.id ||
            null,



        tool:

            step.tool ||
            null,



        status:

            step.status ||
            "UNKNOWN",



        result:

            step.result ||
            null,



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

            ? "RESULT_COMPLETED"

            : "RESULT_FAILED",

        {

            status:

                result.status,


            verified:

                result.verified

        }

    );









    trace.result =
        result;









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






    collectExecutionMeta(

        trace,

        result

    );







    return trace;


}









/*
 * =========================================================
 * COLLECT EXPERIENCE
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
        !experience
    ) {

        return;

    }








    if (
        experience.used === true
    ) {


        trace.experienceUsage.used =
            true;


    }








    if (
        Array.isArray(
            experience.skills
        )
    ) {


        for (
            const skill
            of experience.skills
        ) {


            const exists =

                trace.experienceUsage.skills
                    .some(

                        item =>

                            JSON.stringify(
                                item
                            )
                            ===
                            JSON.stringify(
                                skill
                            )

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
 * EXECUTION META
 * =========================================================
 */


function collectExecutionMeta(

    trace,

    result

) {


    const meta =
        result.executionMeta;



    if (
        !meta
    ) {

        return;

    }






    if (
        meta.traceId
    ) {


        trace.id =
            meta.traceId;

    }









    const experience =
        meta.experience;



    if (
        experience?.used === true
    ) {


        trace.experienceUsage.used =
            true;


    }



}









/*
 * =========================================================
 * SUMMARY
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
            const result
            of summary.results
        ) {


            updateTraceFromResult(

                trace,

                result

            );


        }


    }





    return trace;


}









/*
 * =========================================================
 * REPLAN EVENT
 * =========================================================
 */


export function addTraceReplan(

    trace,

    data = {}

) {


    if (
        !trace
    ) {

        return trace;

    }





    trace.replans.push({

        ...data,


        timestamp:

            new Date()
                .toISOString()


    });






    addTraceEvent(

        trace,

        "REPLAN",

        data

    );



    return trace;


}









/*
 * =========================================================
 * FINISH
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







    if (
        trace.result?.success === true
    ) {


        trace.status =
            "COMPLETED";


    }

    else if (
        trace.result?.status ===
        "NEEDS_CLARIFICATION"
    ) {


        trace.status =
            "NEEDS_CLARIFICATION";


    }

    else {


        trace.status =
            "FAILED";


    }








    trace.learningReady =
        true;



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



        status:

            trace.status,



        success:

            trace.status ===
            "COMPLETED",



        experience:

            trace.experienceUsage,



        steps:

            trace.steps,



        failures:

            trace.failures,



        replans:

            trace.replans,



        events:

            trace.events,



        result:

            trace.result


    };


}
