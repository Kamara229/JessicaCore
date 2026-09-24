/*
 * =========================================================
 * JESSICA COMPLEX TASK RESPONSE BUILDER
 * =========================================================
 *
 * Формирование ответа для сложной задачи.
 *
 *
 * Ответственность:
 *
 * - объединить результаты;
 * - определить статус ответа;
 * - подготовить API структуру.
 *
 *
 * НЕ содержит:
 *
 * - выполнение;
 * - Planner;
 * - Tools;
 * - Validation;
 * - Experience;
 * - Learning.
 *
 * =========================================================
 */


import {
    composeComplexAnswer
} from "../complexAnswerComposer.js";





/*
 * =========================================================
 * BUILD COMPLEX RESPONSE
 * =========================================================
 */


export async function buildComplexTaskResponse(
    originalTask,
    decomposition,
    subtaskRunResult,
    executionTrace = null
) {



    if (
        !subtaskRunResult ||
        typeof subtaskRunResult !== "object"
    ) {


        return {

            success:false,

            stage:"response",

            text:
                "Нет результатов выполнения задачи.",

            engine:
                "jessica-core",

            mode:
                "complex",

            decomposition,

            executionTrace

        };

    }





    /*
     * =====================================================
     * COMPOSE
     * =====================================================
     */


    let composed;



    try {


        composed =
            await composeComplexAnswer(

                originalTask,

                decomposition,

                subtaskRunResult

            );


    } catch(error) {


        console.error(
            "Complex answer composer error:",
            error
        );


        composed = {

            text:
                "Не удалось сформировать итоговый ответ.",

            source:
                "error"

        };


    }







    /*
     * =====================================================
     * SUMMARY
     * =====================================================
     */


    const summary = {


        total:
            subtaskRunResult.total || 0,


        completed:
            subtaskRunResult.completed || 0,


        needsClarification:
            subtaskRunResult.needsClarification || 0,


        failed:
            subtaskRunResult.failed || 0


    };






    const partial =

        summary.completed > 0 &&

        (
            summary.failed > 0 ||
            summary.needsClarification > 0
        );







    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (
        summary.completed > 0
    ) {


        return {


            success:true,


            text:
                composed.text,


            engine:
                "jessica-core",


            mode:
                "complex",


            partial,


            summary,


            decomposition,


            subtasks:
                subtaskRunResult.results || [],


            answerSource:
                composed.source || "unknown",


            executionTrace


        };

    }








    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    if (
        summary.needsClarification > 0 &&
        summary.failed === 0
    ) {


        return {


            success:false,


            needsClarification:true,


            text:
                composed.text,


            engine:
                "jessica-core",


            mode:
                "complex",


            summary,


            decomposition,


            subtasks:
                subtaskRunResult.results || [],


            executionTrace


        };


    }








    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {


        success:false,


        needsClarification:
            summary.needsClarification > 0,


        text:
            composed.text,


        engine:
            "jessica-core",


        mode:
            "complex",


        summary,


        decomposition,


        subtasks:
            subtaskRunResult.results || [],


        executionTrace


    };


}
