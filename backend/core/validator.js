import {
    validateBasicResult
} from "./validator/basicResultValidator.js";

import {
    validateEvidenceResult
} from "./validator/evidenceResultValidator.js";

import {
    validateWithAI
} from "./validator/aiResultValidator.js";


/*
 * =========================================================
 * JESSICA RESULT VALIDATOR
 * =========================================================
 *
 * Главный координатор проверки результата.
 *
 * Детальная логика вынесена в:
 *
 * core/validator/
 */


export async function validateResult(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    /*
     * =====================================================
     * 1. BASIC VALIDATION
     * =====================================================
     */


    const basic =
        validateBasicResult(
            taskRunResult,
            answerResult
        );


    if (
        basic.valid !== true
    ) {

        return {
            success: true,

            valid: false,

            shouldRetry:
                basic.shouldRetry === true,

            needsClarification:
                basic.needsClarification === true,

            reason:
                basic.reason || ""
        };

    }


    /*
     * =====================================================
     * 2. EVIDENCE VALIDATION
     * =====================================================
     *
     * Проверяем, что Planner не просто запросил
     * доказательства, а TaskRunner реально их получил.
     */


    const evidence =
        validateEvidenceResult(
            plan,
            taskRunResult
        );


    if (
        evidence.valid !== true
    ) {

        return {
            success: true,

            valid: false,

            shouldRetry:
                evidence.shouldRetry === true,

            needsClarification: false,

            reason:
                evidence.reason || ""
        };

    }


    /*
     * =====================================================
     * 3. DIRECT TOOL RESULT
     * =====================================================
     *
     * Если ответ сформирован непосредственно
     * успешным инструментом, дополнительная AI-проверка
     * пока не обязательна.
     *
     * Например current_time.
     */


    if (
        answerResult?.source === "tool"
    ) {

        return {
            success: true,

            valid: true,

            shouldRetry: false,

            needsClarification: false,

            reason:
                "Ответ получен напрямую от успешно выполненного инструмента"
        };

    }


    /*
     * =====================================================
     * 4. AI SEMANTIC VALIDATION
     * =====================================================
     */


    const aiValidation =
        await validateWithAI(
            task,
            plan,
            taskRunResult,
            answerResult
        );


    /*
     * Если AI Validator успешно отработал,
     * используем его решение.
     */


    if (
        aiValidation.success === true
    ) {

        return {
            success: true,

            valid:
                aiValidation.valid === true,

            shouldRetry:
                aiValidation.shouldRetry === true,

            needsClarification:
                aiValidation.needsClarification === true,

            reason:
                aiValidation.reason || ""
        };

    }


    /*
     * =====================================================
     * 5. AI VALIDATOR UNAVAILABLE
     * =====================================================
     *
     * Пока не блокируем ответ только из-за временной
     * недоступности AI Validator, если:
     *
     * - базовая проверка прошла;
     * - evidence реально получено.
     *
     * Позже в Jessica 4.0 здесь появится более строгая
     * политика и автоматическое перепланирование.
     */


    return {
        success: true,

        valid: true,

        shouldRetry: false,

        needsClarification: false,

        reason:
            aiValidation.reason ||
            "AI Validator недоступен, техническая проверка пройдена"
    };

}
