/*
 * =========================================================
 * JESSICA EXECUTION TRACE
 * =========================================================
 *
 * История выполнения одной задачи Jessica.
 *
 *
 * Используется для:
 *
 * - debugging;
 * - analytics;
 * - Learning;
 * - quality control.
 *
 *
 * Этот модуль НЕ:
 *
 * - выполняет задачи;
 * - вызывает Planner;
 * - вызывает Tools;
 * - работает с Experience;
 * - формирует ответы.
 *
 *
 * Его задача:
 *
 * собрать полный execution history.
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



        stats: {


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
 * UPDATE SINGLE RESULT
 * =========================================================
 *
 * Используется для:
 *
 * executeSubtask()
 *
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
 * UPDATE COMPLEX SUMMARY
 * =========================================================
 *
 * Используется после:
 *
 * runSubtasks()
 *
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





    /*
     * Сохраняем результаты всех подзадач
     */


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


        }


    }





    /*
     * Обновляем статистику
     */


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
            result.validated === true



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


            trace.usedTools.push(
                tool
            );


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
        stats.completed > 0 &&
        stats.clarification > 0
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
        stats.completed === 0 &&
        stats.clarification > 0
    ) {


        trace.status =
            "NEEDS_CLARIFICATION";


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
        new Date().toISOString();



    updateStatus(
        trace
    );



    return trace;


}
