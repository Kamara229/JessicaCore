/*
 * =========================================================
 * JESSICA RESULT VALIDATOR
 * =========================================================
 *
 * Главный координатор проверки результата.
 *
 *
 * Цепочка:
 *
 * Result
 *   ↓
 * Basic Validator
 *   ↓
 * Evidence Validator
 *   ↓
 * Source Content Validator
 *   ↓
 * Claim Evidence Validator
 *   ↓
 * AI Semantic Validator
 *
 *
 * Каждый слой отвечает только за свою область.
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
            validation.valid === true,

        shouldRetry:
            validation.shouldRetry === true,

        needsClarification:
            validation.needsClarification === true,

        reason:
            validation.reason || ""

    };

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
        JSON.stringify(basic)
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
        JSON.stringify(evidence)
    );



    if (
        evidence.valid !== true
    ) {

        return buildValidationResult(
            evidence
        );

    }



    /*
     * =====================================================
     * 3. SOURCE CONTENT
     * =====================================================
     *
     * Только если нужен source_content.
     */


    const sourceContent =
        await validateSourceContent(

            task,

            plan,

            taskRunResult

        );


    console.log(
        "Jessica Validator source content:",
        JSON.stringify(sourceContent)
    );



    if (
        sourceContent.success === true &&
        sourceContent.valid !== true
    ) {

        return buildValidationResult(
            sourceContent
        );

    }



    /*
     * =====================================================
     * 4. CLAIM EVIDENCE
     * =====================================================
     *
     * Проверяем факты ответа
     * по реально загруженному источнику.
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
        JSON.stringify(claims)
    );



    if (
        claims.success === true &&
        claims.valid !== true
    ) {

        return buildValidationResult(
            claims
        );

    }



    /*
     * =====================================================
     * 5. TOOL DIRECT ANSWER
     * =====================================================
     */


    if (
        answerResult?.source === "tool"
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
                "Ответ получен напрямую от инструмента"

        };

    }



    /*
     * =====================================================
     * 6. AI SEMANTIC VALIDATOR
     * =====================================================
     */


    const aiValidation =
        await validateWithAI(

            task,

            plan,

            taskRunResult,

            answerResult

        );


    console.log(
        "Jessica Validator AI:",
        JSON.stringify(aiValidation)
    );



    if (
        aiValidation.success === true
    ) {

        return buildValidationResult(
            aiValidation
        );

    }



    /*
     * =====================================================
     * 7. AI UNAVAILABLE FALLBACK
     * =====================================================
     */


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
            (
                "AI semantic validation skipped: " +
                (
                    aiValidation.reason ||
                    "validator unavailable"
                )
            )

    };

}
