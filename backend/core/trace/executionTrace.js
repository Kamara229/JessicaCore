/*
 * =========================================================
 * JESSICA EXECUTION TRACE
 * =========================================================
 *
 * История выполнения Jessica.
 *
 * Используется:
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
 * - знает Planner;
 * - знает Experience;
 * - знает Tools.
 *
 * Он только собирает историю.
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


        task,


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
 * ADD SUBTASK RESULT
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



    trace.subtasks.push({

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

    });



    addTools(
        trace,
        result.usedTools
    );



    addValidationErrors(
        trace,
        result.validationErrors
    );



    updateStats(
        trace,
        result
    );



    updateStatus(
        trace
    );



    return trace;

}





/*
 * =========================================================
 * ADD COMPLEX RESULT
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



    trace.stats = {


        total:
            summary.total || 0,


        completed:
            summary.completed || 0,


        failed:
            summary.failed || 0,


        clarification:
            summary.needsClarification || 0


    };



    updateStatus(
        trace
    );



    return trace;

}





/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


function addTools(
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


function addValidationErrors(
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
    result
) {


    trace.stats.total++;



    switch(
        result.status
    ) {


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
        stats.failed > 0 &&
        stats.completed > 0
    ) {

        trace.status =
            "PARTIAL";

        return;

    }



    if (
        stats.failed > 0
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
        new Date().toISOString();



    if (
        trace.status === "RUNNING"
    ) {

        updateStatus(
            trace
        );

    }



    return trace;

}
