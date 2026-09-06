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
 * JESSICA RESULT VALIDATOR
 * =========================================================
 *
 * Главный координатор проверки результата.
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

        const result = {
            success: true,
            valid: false,
            shouldRetry:
                basic.shouldRetry === true,
            needsClarification:
                basic.needsClarification === true,
            reason:
                basic.reason || ""
        };


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 2. EVIDENCE EXISTS
     * =====================================================
     */

    const evidence =
        validateEvidenceResult(
            plan,
            taskRunResult
        );


    console.log(
        "Jessica Validator evidence:",
        JSON.stringify({
            requiredMode:
                plan?.evidence?.mode || "none",
            ...evidence
        })
    );


    if (
        evidence.valid !== true
    ) {

        const result = {
            success: true,
            valid: false,
            shouldRetry:
                evidence.shouldRetry === true,
            needsClarification: false,
            reason:
                evidence.reason || ""
        };


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 3. SOURCE CONTENT QUALITY
     * =====================================================
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

        const result = {
            success: true,
            valid: false,
            shouldRetry:
                sourceContent.shouldRetry === true,
            needsClarification: false,
            reason:
                sourceContent.reason || ""
        };


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 4. CLAIM EVIDENCE
     * =====================================================
     *
     * Проверяем уже конкретные факты
     * итогового ответа по реальному content.
     */

    const claimEvidence =
        await validateClaimEvidence(
            task,
            plan,
            taskRunResult,
            answerResult
        );


    console.log(
        "Jessica Validator claims:",
        JSON.stringify(claimEvidence)
    );


    if (
        claimEvidence.success === true &&
        claimEvidence.valid !== true
    ) {

        const result = {
            success: true,
            valid: false,
            shouldRetry:
                claimEvidence.shouldRetry === true,
            needsClarification: false,
            reason:
                claimEvidence.reason || ""
        };


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 5. DIRECT TOOL RESULT
     * =====================================================
     */

    if (
        answerResult?.source === "tool"
    ) {

        const result = {
            success: true,
            valid: true,
            shouldRetry: false,
            needsClarification: false,
            reason:
                "Ответ получен напрямую от успешно выполненного инструмента"
        };


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 6. AI ANSWER VALIDATION
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

        const result = {
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


        console.log(
            "Jessica Validator final:",
            JSON.stringify(result)
        );


        return result;
    }


    /*
     * =====================================================
     * 7. AI UNAVAILABLE
     * =====================================================
     */

    const result = {
        success: true,
        valid: true,
        shouldRetry: false,
        needsClarification: false,
        reason:
            aiValidation.reason ||
            "AI Validator недоступен, технические проверки пройдены"
    };


    console.log(
        "Jessica Validator final:",
        JSON.stringify(result)
    );


    return result;
}
