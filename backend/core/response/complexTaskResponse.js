/*
 * =========================================================
 * JESSICA COMPLEX TASK RESPONSE BUILDER
 * =========================================================
 *
 * Формирование финального ответа Jessica
 * для сложной задачи с несколькими подзадачами.
 *
 *
 * Отвечает только за:
 *
 * - объединение результатов;
 * - определение partial результата;
 * - формирование API ответа.
 *
 *
 * НЕ содержит:
 *
 * - выполнение подзадач;
 * - Planner;
 * - Tools;
 * - Validator;
 * - Experience;
 * - Learning.
 *
 *
 * =========================================================
 */


import {
    composeComplexAnswer
} from "../complexAnswerComposer.js";



/*
 * =========================================================
 * BUILD COMPLEX TASK RESPONSE
 * =========================================================
 */


export async function buildComplexTaskResponse(
    originalTask,
    decomposition,
    subtaskRunResult
) {


    /*
     * Защита от некорректных данных
     */


    if (
        !subtaskRunResult ||
        typeof subtaskRunResult !== "object"
    ) {

        return {

            success:
                false,

            text:
                "Jessica не получила результаты выполнения подзадач.",

            engine:
                "jessica-core",

            mode:
                "complex",

            stage:
                "response",

            decomposition

        };

    }


    /*
     * =====================================================
     * COMPOSE ANSWER
     * =====================================================
     */


    const composed =
        await composeComplexAnswer(
            originalTask,
            decomposition,
            subtaskRunResult
        );



    const completed =
        subtaskRunResult.completed || 0;


    const needsClarification =
        subtaskRunResult.needsClarification || 0;


    const failed =
        subtaskRunResult.failed || 0;


    const total =
        subtaskRunResult.total || 0;



    /*
     * =====================================================
     * PARTIAL RESULT
     * =====================================================
     *
     * Есть выполненные подзадачи,
     * но часть требует уточнения
     * или завершилась ошибкой.
     *
     * =====================================================
     */


    const partial =
        completed > 0 &&
        (
            needsClarification > 0 ||
            failed > 0
        );



    const summary = {

        total,

        completed,

        needsClarification,

        failed

    };



    /*
     * =====================================================
     * SUCCESS WITH RESULTS
     * =====================================================
     *
     * Если хотя бы одна подзадача выполнена,
     * отдаём полезный результат.
     *
     * =====================================================
     */


    if (
        completed > 0
    ) {

        return {

            success:
                true,

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
                subtaskRunResult.results ||
                [],

            answerSource:
                composed.source ||
                "unknown"

        };

    }



    /*
     * =====================================================
     * ONLY CLARIFICATION REQUIRED
     * =====================================================
     */


    if (
        needsClarification > 0 &&
        failed === 0
    ) {

        return {

            success:
                false,

            needsClarification:
                true,

            text:
                composed.text,

            engine:
                "jessica-core",

            mode:
                "complex",

            summary,

            decomposition,

            subtasks:
                subtaskRunResult.results ||
                []

        };

    }



    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {

        success:
            false,

        needsClarification:
            needsClarification > 0,

        text:
            composed.text,

        engine:
            "jessica-core",

        mode:
            "complex",

        summary,

        decomposition,

        subtasks:
            subtaskRunResult.results ||
            []

    };


}
