/*
 * =========================================================
 * JESSICA EXECUTION TRACE
 * =========================================================
 *
 * История выполнения задачи Jessica.
 *
 *
 * Используется:
 *
 * - debugging;
 * - analytics;
 * - Learning;
 * - quality control.
 *
 *
 * Хранит:
 *
 * - выполнение;
 * - инструменты;
 * - ошибки;
 * - Experience usage.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает Planner;
 * - изменяет Skills.
 *
 * =========================================================
 */







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
            createTraceId(),



        task:
            task || "",



        startedAt:
            new Date().toISOString(),



        finishedAt:
            null,



        status:
            "RUNNING",




        subtasks:
            [],



        usedTools:
            [],



        validationErrors:
            [],






        /*
         * Память Jessica,
         * использованная при выполнении
         */


        experienceUsage:

        {


            used:
                false,



            skills:
                [],



            successfulUses:
                0,



            failedUses:
                0


        },







        stats:

        {


            total:
                0,


            completed:
                0,


            failed:
                0,


            clarification:
                0


        }


    };

}









/*
 * =========================================================
 * TRACE ID
 * =========================================================
 */


function createTraceId() {


    try {


        if (
            typeof crypto !== "undefined" &&
            crypto.randomUUID
        ) {

            return crypto.randomUUID();

        }


    } catch(error) {

    }



    return (

        Date.now()
        +
        "-"
        +
        Math.random()
            .toString(36)
            .substring(2)

    );


}









/*
 * =========================================================
 * UPDATE RESULT
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





    trace.subtasks.push(

        normalizeSubtaskResult(
            result
        )

    );






    collectTools(

        trace,

        result.usedTools

    );





    collectValidationErrors(

        trace,

        result.validationErrors

    );





    collectExperienceUsage(

        trace,

        result

    );





    updateStats(

        trace,

        result.status

    );





    updateStatus(

        trace

    );





    return trace;


}









/*
 * =========================================================
 * EXPERIENCE COLLECTION
 * =========================================================
 */


function collectExperienceUsage(

    trace,

    result

) {


    const experience =

        result?.executionMeta?.experience;



    if (
        !experience ||
        experience.used !== true
    ) {

        return;

    }






    trace.experienceUsage.used =
        true;






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

                trace.experienceUsage.skills.some(

                    item =>

                        (

                            item?.id ||
                            item

                        )

                        ===

                        (

                            skill?.id ||
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






    if (
        result.status === "COMPLETED"
    ) {


        trace.experienceUsage.successfulUses++;


    }


    if (
        result.status === "FAILED"
    ) {


        trace.experienceUsage.failedUses++;


    }


}









/*
 * =========================================================
 * SUMMARY UPDATE
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





    if (
        Array.isArray(
            summary.results
        )
    ) {


        trace.subtasks =

            summary.results.map(

                item =>

                    normalizeSubtaskResult(
                        item
                    )

            );





        for (
            const item
            of summary.results
        ) {


            collectTools(

                trace,

                item.usedTools

            );



            collectValidationErrors(

                trace,

                item.validationErrors

            );



            collectExperienceUsage(

                trace,

                item

            );


        }


    }






    trace.stats = {


        total:

            Number(
                summary.total || 0
            ),



        completed:

            Number(
                summary.completed || 0
            ),



        failed:

            Number(
                summary.failed || 0
            ),



        clarification:

            Number(
                summary.needsClarification || 0
            )


    };





    updateStatus(
        trace
    );



    return trace;


}









/*
 * =========================================================
 * NORMALIZE RESULT
 * =========================================================
 */


function normalizeSubtaskResult(
    result
) {


    return {


        id:
            result.id || null,



        status:
            result.status || "UNKNOWN",



        stage:
            result.stage || null,



        result:
            result.result || "",



        validated:
            result.validated === true,



        experienceUsed:

            result?.executionMeta?.experience?.used === true


    };

}









/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


function collectTools(

    trace,

    tools

) {


    if (
        !Array.isArray(tools)
    ) {

        return;

    }



    for (
        const tool
        of tools
    ) {


        if (
            !trace.usedTools.includes(tool)
        ) {


            trace.usedTools.push(tool);


        }

    }


}









/*
 * =========================================================
 * VALIDATION ERRORS
 * =========================================================
 */


function collectValidationErrors(

    trace,

    errors

) {


    if (
        !Array.isArray(errors)
    ) {

        return;

    }



    trace.validationErrors.push(
        ...errors
    );

}









/*
 * =========================================================
 * STATS
 * =========================================================
 */


function updateStats(

    trace,

    status

) {


    trace.stats.total++;



    switch(status) {


        case "COMPLETED":

            trace.stats.completed++;

            break;



        case "FAILED":

            trace.stats.failed++;

            break;



        case "NEEDS_CLARIFICATION":

            trace.stats.clarification++;

            break;


    }


}









/*
 * =========================================================
 * STATUS
 * =========================================================
 */


function updateStatus(
    trace
) {


    const stats =
        trace.stats;



    if (
        stats.completed > 0 &&
        stats.failed > 0
    ) {

        trace.status =
            "PARTIAL";

        return;

    }



    if (
        stats.failed > 0 &&
        stats.completed === 0
    ) {

        trace.status =
            "FAILED";

        return;

    }



    if (
        stats.total > 0 &&
        stats.completed === stats.total
    ) {

        trace.status =
            "COMPLETED";

        return;

    }



    trace.status =
        "RUNNING";


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



    updateStatus(
        trace
    );



    return trace;


}
