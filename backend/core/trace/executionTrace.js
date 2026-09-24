/*
 * =========================================================
 * JESSICA EXECUTION TRACE
 * =========================================================
 *
 * История выполнения одной задачи.
 *
 * Используется для:
 *
 * - отладки;
 * - анализа ошибок;
 * - обучения Jessica;
 * - контроля качества ответов.
 *
 * =========================================================
 */


export function createExecutionTrace(
    task
) {

    return {

        id:
            crypto.randomUUID
                ? crypto.randomUUID()
                : Date.now().toString(),


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
 * UPDATE TRACE
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



    /*
     * сохраняем результат подзадачи
     */

    trace.subtasks.push(
        {

            status:
                result.status || "UNKNOWN",


            result:
                result.result || "",


            stage:
                result.stage || null,


            validated:
                result.validated === true


        }
    );



    /*
     * инструменты
     */

    if (
        Array.isArray(
            result.usedTools
        )
    ) {

        trace.usedTools.push(
            ...result.usedTools
        );

    }



    /*
     * ошибки валидации
     */

    if (
        Array.isArray(
            result.validationErrors
        )
    ) {

        trace.validationErrors.push(
            ...result.validationErrors
        );

    }



    /*
     * статистика
     */

    trace.stats.total++;


    if (
        result.status === "COMPLETED"
    ) {

        trace.stats.completed++;

    }


    else if (
        result.status === "FAILED"
    ) {

        trace.stats.failed++;

    }


    else if (
        result.status === "NEEDS_CLARIFICATION"
    ) {

        trace.stats.clarification++;

    }



    /*
     * общий статус
     */

    if (
        trace.stats.failed > 0
    ) {

        trace.status =
            "PARTIAL";

    }


    if (
        trace.stats.total > 0 &&
        trace.stats.completed === trace.stats.total
    ) {

        trace.status =
            "COMPLETED";

    }



    return trace;

}



/*
 * =========================================================
 * FINISH TRACE
 * =========================================================
 */


export function finishExecutionTrace(
    trace
) {


    if (!trace) {
        return trace;
    }


    trace.finishedAt =
        new Date().toISOString();



    return trace;

}
