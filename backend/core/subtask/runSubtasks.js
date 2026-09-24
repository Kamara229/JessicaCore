import {
    executeSubtask
} from "./executeSubtask.js";


import {
    buildSubtaskSummary
} from "./subtaskSummary.js";



/*
 * =========================================================
 * JESSICA RUN SUBTASKS
 * =========================================================
 *
 * Последовательный запуск нескольких подзадач.
 *
 *
 * Flow:
 *
 * decomposition
 *      ↓
 * subtasks[]
 *      ↓
 * executeSubtask()
 *      ↓
 * results[]
 *      ↓
 * summary
 *
 *
 * НЕ отвечает за:
 *
 * - Planner;
 * - Experience;
 * - Tools;
 * - Learning;
 * - финальный ответ.
 *
 * =========================================================
 */



/*
 * =========================================================
 * FALLBACK FAILED RESULT
 * =========================================================
 */


function buildUnhandledErrorResult(
    subtask,
    error
) {


    return {

        id:
            subtask?.id ?? null,


        text:
            typeof subtask?.text === "string"
                ? subtask.text
                : "",


        success:
            false,


        status:
            "FAILED",


        stage:
            "subtask",


        result:
            "Непредвиденная ошибка выполнения подзадачи",



        error:
            error?.message ||
            "unknown error",



        executionMeta: {

            failedBeforeExecution:
                true

        }

    };

}



/*
 * =========================================================
 * RUN ALL SUBTASKS
 * =========================================================
 */


export async function runSubtasks(
    decomposition
) {


    const subtasks =
        Array.isArray(
            decomposition?.subtasks
        )
            ? decomposition.subtasks
            : [];



    /*
     * =====================================================
     * EMPTY
     * =====================================================
     */


    if (
        subtasks.length === 0
    ) {

        return buildSubtaskSummary(
            []
        );

    }



    const results =
        [];



    /*
     * =====================================================
     * SEQUENTIAL EXECUTION
     * =====================================================
     *
     * Пока оставляем последовательный запуск.
     *
     * Причины:
     *
     * - контроль API лимитов;
     * - независимый Experience;
     * - проще анализировать ошибки;
     * - стабильнее для Learning.
     *
     * =====================================================
     */


    for (
        const subtask
        of subtasks
    ) {


        console.log(

            `Jessica subtask ${subtask?.id}:`,

            subtask?.text

        );



        try {


            const result =
                await executeSubtask(
                    subtask
                );


            results.push(
                result
            );



        } catch (error) {


            console.error(

                `Unhandled subtask error ${subtask?.id}:`,

                error

            );



            results.push(

                buildUnhandledErrorResult(
                    subtask,
                    error
                )

            );

        }


    }



    /*
     * =====================================================
     * SUMMARY
     * =====================================================
     *
     * buildSubtaskSummary отвечает
     * только за статистику.
     *
     * =====================================================
     */


    const summary =
        buildSubtaskSummary(
            results
        );



    /*
     * =====================================================
     * TRACE DATA
     * =====================================================
     *
     * Передаём результаты выше.
     *
     * Jessica Core уже решает,
     * как использовать trace.
     *
     * =====================================================
     */


    return {

        ...summary,


        results,


        executionTraces:

            results
                .map(
                    item =>
                        item.executionTrace || null
                )
                .filter(
                    Boolean
                )

    };


}
