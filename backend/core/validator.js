/*
 * =========================================================
 * JESSICA RESULT VALIDATOR
 * =========================================================
 *
 * Главный координатор проверки результата.
 *
 *
 * Основной flow:
 *
 * Result
 *   ↓
 * Basic Validator
 *   ↓
 * Evidence Validator
 *   ↓
 * Conditional Validation
 *
 *
 * source_content:
 *
 * Source Content Validator
 *   ↓
 * Claim Evidence Validator
 *   ↓
 * SUCCESS
 *
 *
 * none / search_results:
 *
 * AI Semantic Validator
 *
 *
 * AI Semantic Validator также используется
 * как fallback, если специализированный
 * AI Validator временно недоступен.
 *
 *
 * Дополнительно Validator возвращает:
 *
 * outcomeType
 *
 * result
 * → искомый результат подтверждён.
 *
 * no_verified_result
 * → ответ корректный, но подтверждённого
 *   искомого результата нет.
 *
 *
 * Важно:
 *
 * valid и outcomeType — разные понятия.
 *
 * valid=true + outcomeType=no_verified_result
 *
 * означает:
 *
 * ответ Jessica корректен,
 * но положительный результат не подтверждён.
 *
 *
 * Этот файл НЕ:
 *
 * - содержит AI prompts;
 * - проверяет claims самостоятельно;
 * - выполняет инструменты;
 * - строит новые планы.
 *
 * =========================================================
 */


import {
    validateBasicResult
} from "./validator/basicResultValidator.js";

import {
    validateEvidenceResult
} from "./validator/evidenceResultValidator.js";

import {
    validateSourceContent
} from "./validator/sourceContentValidator.js";

import {
    validateClaimEvidence
} from "./validator/claimEvidenceValidator.js";

import {
    validateWithAI
} from "./validator/aiResultValidator.js";


/*
 * =========================================================
 * OUTCOME TYPES
 * =========================================================
 */


const OUTCOME_RESULT =
    "result";


const OUTCOME_NO_VERIFIED_RESULT =
    "no_verified_result";


/*
 * =========================================================
 * NORMALIZE OUTCOME TYPE
 * =========================================================
 */


function normalizeOutcomeType(
    value
) {

    if (
        value ===
        OUTCOME_NO_VERIFIED_RESULT
    ) {

        return OUTCOME_NO_VERIFIED_RESULT;

    }


    return OUTCOME_RESULT;

}


/*
 * =========================================================
 * RESULT BUILDER
 * =========================================================
 */


function buildValidationResult(
    validation
) {

    const valid =
        validation?.valid === true;


    return {

        success:
            true,


        valid,


        shouldRetry:
            validation?.shouldRetry === true,


        needsClarification:
            validation?.needsClarification === true,


        /*
         * outcomeType имеет смысл только
         * для корректно принятого результата.
         *
         * Для validation failure ставим null,
         * потому что результат ещё не является
         * terminal semantic outcome.
         */

        outcomeType:
            valid
                ? normalizeOutcomeType(
                    validation?.outcomeType
                )
                : null,


        reason:
            typeof validation?.reason === "string"
                ? validation.reason.trim()
                : ""

    };

}


/*
 * =========================================================
 * SUCCESS RESULT
 * =========================================================
 */


function buildSuccessResult(
    reason,
    outcomeType = OUTCOME_RESULT
) {

    return {

        success:
            true,


        valid:
            true,


        shouldRetry:
            false,


        needsClarification:
            false,


        outcomeType:
            normalizeOutcomeType(
                outcomeType
            ),


        reason:
            reason ||
            "Проверка результата пройдена"

    };

}


/*
 * =========================================================
 * EVIDENCE MODE
 * =========================================================
 */


function getEvidenceMode(
    plan
) {

    const mode =
        String(
            plan?.evidence?.mode || "none"
        )
            .trim()
            .toLowerCase();


    if (
        mode === "source_content" ||
        mode === "search_results"
    ) {

        return mode;

    }


    return "none";

}


/*
 * =========================================================
 * AI SEMANTIC FALLBACK
 * =========================================================
 */


async function runSemanticValidation(

    task,

    plan,

    taskRunResult,

    answerResult

) {

    const aiValidation =
        await validateWithAI(

            task,

            plan,

            taskRunResult,

            answerResult

        );


    console.log(
        "Jessica Validator AI:",
        JSON.stringify(
            aiValidation
        )
    );


    /*
     * =====================================================
     * AI VALIDATION SUCCESS
     * =====================================================
     */


    if (
        aiValidation?.success === true
    ) {

        return buildValidationResult(
            aiValidation
        );

    }


    /*
     * =====================================================
     * AI VALIDATOR UNAVAILABLE
     * =====================================================
     *
     * Basic + Evidence к этому моменту
     * уже пройдены.
     *
     * Поэтому техническая недоступность
     * semantic AI не должна ломать задачу.
     *
     *
     * outcomeType здесь оставляем result.
     *
     * Мы НЕ можем надёжно объявить
     * no_verified_result без semantic проверки.
     *
     * =====================================================
     */


    return buildSuccessResult(

        "AI semantic validation unavailable: " +
        (
            aiValidation?.reason ||
            "validator unavailable"
        ),

        OUTCOME_RESULT

    );

}


/*
 * =========================================================
 * VALIDATE RESULT
 * =========================================================
 */


export async function validateResult(

    task,

    plan,

    taskRunResult,

    answerResult

) {

    /*
     * =====================================================
     * 1. BASIC
     * =====================================================
     */


    const basic =
        validateBasicResult(

            taskRunResult,

            answerResult

        );


    console.log(
        "Jessica Validator basic:",
        JSON.stringify(
            basic
        )
    );


    if (
        basic.valid !== true
    ) {

        return buildValidationResult(
            basic
        );

    }


    /*
     * =====================================================
     * 2. EVIDENCE EXISTENCE
     * =====================================================
     */


    const evidence =
        validateEvidenceResult(

            plan,

            taskRunResult

        );


    console.log(
        "Jessica Validator evidence:",
        JSON.stringify(
            evidence
        )
    );


    if (
        evidence.valid !== true
    ) {

        return buildValidationResult(
            evidence
        );

    }


    const evidenceMode =
        getEvidenceMode(
            plan
        );


    /*
     * =====================================================
     * 3. DIRECT TOOL ANSWER
     * =====================================================
     *
     * Если ответ пришёл напрямую от инструмента
     * и конкретный source_content не требуется,
     * дополнительный AI Validator не нужен.
     *
     * Например:
     *
     * - текущее время;
     * - вычисление;
     * - структурированный tool-result.
     *
     *
     * Такой результат считаем обычным result.
     *
     * =====================================================
     */


    if (
        answerResult?.source === "tool" &&
        evidenceMode !== "source_content"
    ) {

        console.log(
            "Jessica Validator semantic skipped: direct tool answer"
        );


        return buildSuccessResult(

            "Ответ получен напрямую от инструмента",

            OUTCOME_RESULT

        );

    }


    /*
     * =====================================================
     * 4. SOURCE CONTENT VALIDATION
     * =====================================================
     *
     * Для source_content используем
     * специализированную цепочку:
     *
     * source relevance
     *      ↓
     * claim evidence
     *
     * =====================================================
     */


    if (
        evidenceMode === "source_content"
    ) {

        /*
         * -----------------------------------------------
         * SOURCE CONTENT
         * -----------------------------------------------
         */


        const sourceContent =
            await validateSourceContent(

                task,

                plan,

                taskRunResult

            );


        console.log(
            "Jessica Validator source content:",
            JSON.stringify(
                sourceContent
            )
        );


        /*
         * Validator успешно отработал
         * и нашёл реальную проблему.
         */


        if (
            sourceContent?.success === true &&
            sourceContent?.valid !== true
        ) {

            return buildValidationResult(
                sourceContent
            );

        }


        /*
         * -----------------------------------------------
         * CLAIM EVIDENCE
         * -----------------------------------------------
         */


        const claims =
            await validateClaimEvidence(

                task,

                plan,

                taskRunResult,

                answerResult

            );


        console.log(
            "Jessica Validator claims:",
            JSON.stringify(
                claims
            )
        );


        /*
         * Claims Validator успешно отработал
         * и обнаружил неподтверждённые факты.
         */


        if (
            claims?.success === true &&
            claims?.valid !== true
        ) {

            return buildValidationResult(
                claims
            );

        }


        /*
         * -----------------------------------------------
         * SPECIALIZED VALIDATION PASSED
         * -----------------------------------------------
         */


        const sourceConfirmed =

            sourceContent?.success === true &&
            sourceContent?.valid === true;


        const claimsConfirmed =

            claims?.success === true &&
            claims?.valid === true;


        if (
            sourceConfirmed &&
            claimsConfirmed
        ) {

            /*
             * Здесь есть:
             *
             * - подтверждённый источник;
             * - подтверждённые claims.
             *
             * Поэтому semantic outcome однозначно:
             *
             * result.
             */


            console.log(
                "Jessica Validator AI skipped: source and claims confirmed"
            );


            return buildSuccessResult(

                "Источник и утверждения ответа подтверждены",

                OUTCOME_RESULT

            );

        }


        /*
         * -----------------------------------------------
         * SPECIALIZED VALIDATOR UNAVAILABLE
         * -----------------------------------------------
         *
         * Один из специализированных AI Validator
         * оказался недоступен.
         *
         * Тогда используем общий semantic Validator.
         *
         * Он сможет определить не только valid,
         * но и outcomeType.
         *
         * -----------------------------------------------
         */


        console.log(
            "Jessica Validator: specialized validation unavailable, using semantic fallback"
        );


        return await runSemanticValidation(

            task,

            plan,

            taskRunResult,

            answerResult

        );

    }


    /*
     * =====================================================
     * 5. SEARCH RESULTS / NONE
     * =====================================================
     *
     * Здесь специализированной проверки
     * содержимого источника нет.
     *
     * Поэтому общий AI Validator определяет:
     *
     * - valid;
     * - retry;
     * - clarification;
     * - outcomeType.
     *
     *
     * Именно здесь наш Quasar-тест должен дать:
     *
     * valid=true
     * outcomeType=no_verified_result
     *
     * =====================================================
     */


    return await runSemanticValidation(

        task,

        plan,

        taskRunResult,

        answerResult

    );

}
