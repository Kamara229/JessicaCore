/*
 * =========================================================
 * JESSICA COMPLEX TASK RESPONSE BUILDER
 * =========================================================
 *
 * Формирование API ответа
 * для сложной задачи.
 *
 *
 * Ответственность:
 *
 * - объединить результаты;
 * - определить состояние ответа;
 * - вернуть структуру API.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает Planner;
 * - вызывает Tools;
 * - работает с Experience;
 * - работает с Learning.
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

            status:
                "FAILED",

            stage:
                "response",

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
     * COMPOSE ANSWER
     * =====================================================
     */


    let composed = {

        text:
            "",

        source:
            "unknown"

    };



    try {


        const result =
            await composeComplexAnswer(

                originalTask,

                decomposition,

                subtaskRunResult

            );


        if (
            result &&
            typeof result === "object"
        ) {

            composed =
                result;

        }


    } catch(error) {


        console.error(

            "Jessica complex composer error:",

            error

        );


        composed = {

            text:
                "Jessica выполнила часть задачи, но не смогла объединить результаты.",


            source:
                "composer-error"

        };


    }









    /*
     * =====================================================
     * SUMMARY
     * =====================================================
     */


    const summary = {


        total:
            Number(
                subtaskRunResult.total || 0
            ),


        completed:
            Number(
                subtaskRunResult.completed || 0
            ),


        needsClarification:
            Number(
                subtaskRunResult.needsClarification || 0
            ),


        failed:
            Number(
                subtaskRunResult.failed || 0
            )


    };






    const subtasks =
        Array.isArray(
            subtaskRunResult.results
        )
            ? subtaskRunResult.results
            : [];








    /*
     * =====================================================
     * PARTIAL
     * =====================================================
     */


    const partial =

        summary.completed > 0 &&

        (
            summary.failed > 0 ||
            summary.needsClarification > 0
        );









    /*
     * =====================================================
     * COMPLETED / PARTIAL
     * =====================================================
     */


    if (
        summary.completed > 0
    ) {


        return {


            success:true,


            status:
                partial
                    ? "PARTIAL"
                    : "COMPLETED",



            text:
                composed.text ||
                "Задача выполнена.",



            engine:
                "jessica-core",



            mode:
                "complex",



            partial,



            summary,



            decomposition,



            subtasks,



            answerSource:
                composed.source ||
                "unknown",



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


            status:
                "NEEDS_CLARIFICATION",



            needsClarification:true,



            text:
                composed.text ||
                "Требуется уточнение.",



            engine:
                "jessica-core",



            mode:
                "complex",



            summary,



            decomposition,



            subtasks,



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


        status:
            "FAILED",



        needsClarification:
            summary.needsClarification > 0,



        text:
            composed.text ||
            "Jessica не смогла выполнить задачу.",



        engine:
            "jessica-core",



        mode:
            "complex",



        summary,



        decomposition,



        subtasks,



        executionTrace



    };


}
