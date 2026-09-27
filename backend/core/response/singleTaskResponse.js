/*
 * =========================================================
 * JESSICA SINGLE TASK RESPONSE BUILDER v2
 * =========================================================
 *
 * Формирование API ответа Jessica
 * для одной подзадачи.
 *
 *
 * Flow:
 *
 * Execution Result
 *        ↓
 * Single Task Response
 *        ↓
 * Android API Response
 *
 *
 * Ответственность:
 *
 * - преобразовать Execution Result;
 * - сохранить публичный API контракт;
 * - вернуть текст ответа;
 * - прикрепить executionTrace.
 *
 *
 * НЕ:
 *
 * - выполняет Execution;
 * - запускает Planner;
 * - выполняет Tools;
 * - валидирует;
 * - изменяет Experience;
 * - выполняет Learning.
 *
 * =========================================================
 */



/*
 * =========================================================
 * ANSWER TEXT
 * =========================================================
 */


function getAnswerText(
    result
) {

    if (
        typeof result?.answer?.text === "string"
    ) {

        return result.answer.text.trim();

    }


    return "";

}



/*
 * =========================================================
 * FAILURE TEXT
 * =========================================================
 */


function getFailureText(
    result
) {

    if (
        typeof result?.failure?.reason === "string" &&
        result.failure.reason.trim()
    ) {

        return result.failure.reason.trim();

    }


    return "Jessica не смогла выполнить задачу.";

}



/*
 * =========================================================
 * CLARIFICATION TEXT
 * =========================================================
 */


function getClarificationText(
    result
) {

    if (
        typeof result?.clarification?.reason === "string" &&
        result.clarification.reason.trim()
    ) {

        return result.clarification.reason.trim();

    }


    return "Для выполнения задачи требуется уточнение.";

}



/*
 * =========================================================
 * BUILD SINGLE TASK RESPONSE
 * =========================================================
 */


export function buildSingleTaskResponse(
    result,
    decomposition,
    executionTrace = null
) {


    /*
     * =====================================================
     * INVALID RESULT
     * =====================================================
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
                "response",

            executionTrace

        };

    }



    /*
     * =====================================================
     * COMPLETED
     * =====================================================
     */


    if (
        result.status === "COMPLETED"
    ) {

        return {

            success:
                true,

            text:
                getAnswerText(
                    result
                ),

            engine:
                "jessica-core",

            mode:
                "single",

            validated:
                result.verified === true,

            answerSource:
                result?.answer?.source ||
                "unknown",

            decomposition,

            plan:
                result.currentPlan ||
                null,

            toolResults:
                result.toolResults ||
                [],

            usedTools:
                result?.executionMeta?.usedTools ||
                [],

            experience:
                result?.executionMeta?.experience ||
                null,

            validation:
                result.validation ||
                null,

            executionTrace

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
                getClarificationText(
                    result
                ),

            engine:
                "jessica-core",

            mode:
                "single",

            stage:
                result?.clarification?.stage ||
                "subtask",

            decomposition,

            plan:
                result.currentPlan ||
                null,

            toolResults:
                result.toolResults ||
                [],

            usedTools:
                result?.executionMeta?.usedTools ||
                [],

            experience:
                result?.executionMeta?.experience ||
                null,

            executionTrace

        };

    }



    /*
     * =====================================================
     * FAILED / NO VERIFIED RESULT
     * =====================================================
     */


    return {

        success:
            false,

        shouldRetry:
            result?.failure?.shouldRetry === true,

        text:
            getFailureText(
                result
            ),

        engine:
            "jessica-core",

        mode:
            "single",

        stage:
            result?.failure?.stage ||
            "subtask",

        failureType:
            result?.failure?.failureType ||
            null,

        decomposition,

        plan:
            result.currentPlan ||
            null,

        toolResults:
            result.toolResults ||
            [],

        usedTools:
            result?.executionMeta?.usedTools ||
            [],

        experience:
            result?.executionMeta?.experience ||
            null,

        executionTrace

    };

}
