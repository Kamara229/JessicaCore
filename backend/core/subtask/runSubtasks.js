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
 * Выполнение нескольких подзадач Jessica.
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
 * Ответственность:
 *
 * - последовательный запуск;
 * - обработка ошибок;
 * - сбор результатов.
 *
 *
 * НЕ отвечает за:
 *
 * - Planner;
 * - Experience;
 * - Tools;
 * - Response;
 * - Learning;
 * - Trace.
 *
 * =========================================================
 */





/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


function buildFailedSubtaskResult(
    subtask,
    error
) {


    return {


        id:
            subtask?.id || null,



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
            "unknown error"


    };


}







/*
 * =========================================================
 * RUN SUBTASKS
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






    const results = [];






    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     *
     * Последовательный запуск.
     *
     * Позже можно заменить
     * на очередь / worker pool.
     *
     * =====================================================
     */


    for (
        const subtask
        of subtasks
    ) {



        console.log(

            "Jessica subtask:",

            {

                id:
                    subtask?.id,


                text:
                    subtask?.text

            }

        );





        try {


            const result =
                await executeSubtask(
                    subtask
                );



            results.push(
                result
            );



        } catch(error) {



            console.error(

                `Jessica subtask failed ${subtask?.id}:`,

                error

            );



            results.push(

                buildFailedSubtaskResult(
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
     */


    return buildSubtaskSummary(
        results
    );



}
