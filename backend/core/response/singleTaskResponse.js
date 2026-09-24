/*
 * =========================================================
 * JESSICA SINGLE TASK RESPONSE BUILDER
 * =========================================================
 *
 * Формирование ответа Jessica
 * для одной подзадачи.
 *
 *
 * Отвечает только за:
 *
 * - success response;
 * - clarification response;
 * - failed response.
 *
 *
 * НЕ содержит:
 *
 * - выполнение задачи;
 * - Planner;
 * - Tools;
 * - Validator;
 * - Learning;
 * - Experience.
 *
 *
 * =========================================================
 */


/*
 * =========================================================
 * BUILD SINGLE TASK RESPONSE
 * =========================================================
 *
 * Преобразует результат executeSubtask()
 * в стандартный ответ Jessica Core.
 *
 * =========================================================
 */


export function buildSingleTaskResponse(
    result,
    decomposition
) {


    /*
     * Защита от некорректного результата
     */


    if (
        !result ||
        typeof result !== "object"
    ) {

        return {

            success:
                false,

            text:
                "Jessica получила некорректный результат выполнения.",

            engine:
                "jessica-core",

            mode:
                "single",

            stage:
                "response"

        };

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (
        result.status === "COMPLETED"
    ) {

        return {

            success:
                true,

            text:
                result.result,

            engine:
                "jessica-core",

            mode:
                "single",

            validated:
                result.validated === true,

            answerSource:
                result.answerSource ||
                "unknown",

            usedTools:
                result.usedTools ||
                [],

            decomposition,

            plan:
                result.plan ||
                null,

            toolResults:
                result.toolResults ||
                []

        };

    }


    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    if (
        result.status === "NEEDS_CLARIFICATION"
    ) {

        return {

            success:
                false,

            needsClarification:
                true,

            text:
                result.result ||
                "Для выполнения задачи требуется уточнение.",

            engine:
                "jessica-core",

            mode:
                "single",

            stage:
                result.stage ||
                "subtask",

            decomposition,

            plan:
                result.plan ||
                null,

            toolResults:
                result.toolResults ||
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

        shouldRetry:
            result.shouldRetry === true,

        text:
            result.result ||
            "Jessica не смогла выполнить задачу.",

        engine:
            "jessica-core",

        mode:
            "single",

        stage:
            result.stage ||
            "subtask",

        decomposition,

        plan:
            result.plan ||
            null,

        toolResults:
            result.toolResults ||
            []

    };


}
