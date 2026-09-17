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
 * RESULT BUILDER
 * =========================================================
 */


function buildValidationResult(
    validation
) {

    return {

        success:
            true,


        valid:
            validation?.valid === true,


        shouldRetry:
            validation?.shouldRetry === true,


        needsClarification:
            validation?.needsClarification === true,


        reason:
            validation?.reason || ""

    };

}



/*
 * =========================================================
 * SUCCESS RESULT
 * =========================================================
 */


function buildSuccessResult(
    reason
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


        reason:
            reason || "Проверка результата пройдена"

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



    if (
        aiValidation?.success === true
    ) {

        return buildValidationResult(
            aiValidation
        );

    }



    /*
     * AI Validator недоступен.
     *
     * Базовые и evidence-проверки
     * к этому моменту уже пройдены.
     *
     * Поэтому не ломаем выполнение задачи.
     */


    return buildSuccessResult(

        "AI semantic validation unavailable: " +
        (
            aiValidation?.reason ||
            "validator unavailable"
        )

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
     * - время;
     * - вычисление;
     * - структурированный tool-result.
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
            "Ответ получен напрямую от инструмента"
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
         * Validator отработал
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
         * Claims Validator отработал
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
         *
         * Оба специализированных Validator:
         *
         * - подтвердили источник;
         * - подтвердили утверждения ответа.
         *
         * Общий AI Semantic Validator здесь
         * уже не добавляет существенной проверки.
         *
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


            console.log(
                "Jessica Validator AI skipped: source and claims confirmed"
            );


            return buildSuccessResult(
                "Источник и утверждения ответа подтверждены"
            );

        }



        /*
         * Один из специализированных AI Validator
         * оказался недоступен.
         *
         * Тогда используем общий AI Validator
         * как fallback.
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
     * Поэтому используем общий
     * AI Semantic Validator.
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
