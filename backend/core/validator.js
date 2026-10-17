/*
 * =========================================================
 * JESSICA RESULT VALIDATOR v3
 * =========================================================
 *
 * Главный координатор проверки результата.
 *
 * Дополнительно сохраняет failureScope:
 *
 * presentation
 * execution
 * evidence
 * clarification
 *
 * Это позволяет отличать ошибку итогового текста
 * от ошибки Execution Route.
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


const OUTCOME_RESULT =
    "result";


const OUTCOME_NO_VERIFIED_RESULT =
    "no_verified_result";


const MODE_BASIC =
    "basic";


const MODE_TOOL_DIRECT =
    "tool_direct";


const MODE_SEMANTIC =
    "semantic";


const MODE_SPECIALIZED =
    "specialized";


const MODE_SEMANTIC_FALLBACK =
    "semantic_fallback";


const MODE_UNAVAILABLE_FALLBACK =
    "unavailable_fallback";


const FAILURE_SCOPE = {

    PRESENTATION:
        "presentation",

    EXECUTION:
        "execution",

    EVIDENCE:
        "evidence",

    CLARIFICATION:
        "clarification"

};


function normalizeOutcomeType(
    value
) {

    if(
        value ===
        OUTCOME_NO_VERIFIED_RESULT
    ){

        return OUTCOME_NO_VERIFIED_RESULT;

    }


    return OUTCOME_RESULT;

}


function normalizeFailureScope(
    validation,
    fallback
) {

    if(
        validation?.valid === true
    ){

        return null;

    }


    if(
        validation?.needsClarification === true
    ){

        return FAILURE_SCOPE.CLARIFICATION;

    }


    const scope =
        String(
            validation?.failureScope || ""
        )
            .trim()
            .toLowerCase();


    if(
        Object.values(
            FAILURE_SCOPE
        )
            .includes(
                scope
            )
    ){

        return scope;

    }


    return fallback || FAILURE_SCOPE.EXECUTION;

}


function buildValidationResult(

    validation,

    {

        validationMode =
            MODE_BASIC,

        degraded =
            false,

        failureScope =
            FAILURE_SCOPE.EXECUTION

    } = {}

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


        failureScope:
            normalizeFailureScope(
                validation,
                failureScope
            ),


        outcomeType:

            valid

                ? normalizeOutcomeType(
                    validation?.outcomeType
                )

                : null,


        validationMode,


        degraded:
            degraded === true,


        reason:

            typeof validation?.reason === "string"

                ? validation.reason.trim()

                : ""

    };

}


function buildSuccessResult(

    reason,

    outcomeType =
        OUTCOME_RESULT,

    {

        validationMode =
            MODE_BASIC,

        degraded =
            false

    } = {}

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

        failureScope:
            null,

        outcomeType:
            normalizeOutcomeType(
                outcomeType
            ),

        validationMode,

        degraded:
            degraded === true,

        reason:
            reason
            ||
            "Проверка результата пройдена"

    };

}


function getEvidenceMode(
    plan
) {

    const mode =
        String(
            plan?.evidence?.mode || "none"
        )
            .trim()
            .toLowerCase();


    if(
        mode === "source_content"
        ||
        mode === "search_results"
    ){

        return mode;

    }


    return "none";

}


async function runSemanticValidation(

    task,

    plan,

    taskRunResult,

    answerResult,

    {

        fallbackFromSpecialized =
            false

    } = {}

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


    if(
        aiValidation?.success === true
    ){

        return buildValidationResult(

            aiValidation,

            {

                validationMode:

                    fallbackFromSpecialized

                        ? MODE_SEMANTIC_FALLBACK

                        : MODE_SEMANTIC,


                degraded:
                    fallbackFromSpecialized,


                failureScope:
                    FAILURE_SCOPE.EXECUTION

            }

        );

    }


    /*
     * Basic + Evidence уже пройдены.
     * Техническая недоступность AI Validator
     * не должна ломать Execution.
     */

    return buildSuccessResult(

        "AI semantic validation unavailable: "
        +
        (
            aiValidation?.reason
            ||
            "validator unavailable"
        ),

        OUTCOME_RESULT,

        {

            validationMode:
                MODE_UNAVAILABLE_FALLBACK,

            degraded:
                true

        }

    );

}


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


    if(
        basic.valid !== true
    ){

        return buildValidationResult(

            basic,

            {

                validationMode:
                    MODE_BASIC,

                degraded:
                    false,

                failureScope:
                    FAILURE_SCOPE.EXECUTION

            }

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


    if(
        evidence.valid !== true
    ){

        return buildValidationResult(

            evidence,

            {

                validationMode:
                    MODE_BASIC,

                degraded:
                    false,

                failureScope:
                    FAILURE_SCOPE.EVIDENCE

            }

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
     */


    if(
        answerResult?.source === "tool"
        &&
        evidenceMode !== "source_content"
    ){

        console.log(
            "Jessica Validator semantic skipped: direct tool answer"
        );


        return buildSuccessResult(

            "Ответ получен напрямую от инструмента",

            OUTCOME_RESULT,

            {

                validationMode:
                    MODE_TOOL_DIRECT,

                degraded:
                    false

            }

        );

    }


    /*
     * =====================================================
     * 4. SOURCE CONTENT VALIDATION
     * =====================================================
     */


    if(
        evidenceMode === "source_content"
    ){

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


        if(
            sourceContent?.success === true
            &&
            sourceContent?.valid !== true
        ){

            return buildValidationResult(

                sourceContent,

                {

                    validationMode:
                        MODE_SPECIALIZED,

                    degraded:
                        false,

                    failureScope:
                        FAILURE_SCOPE.EVIDENCE

                }

            );

        }


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


        if(
            claims?.success === true
            &&
            claims?.valid !== true
        ){

            return buildValidationResult(

                claims,

                {

                    validationMode:
                        MODE_SPECIALIZED,

                    degraded:
                        false,

                    failureScope:
                        FAILURE_SCOPE.EVIDENCE

                }

            );

        }


        const sourceConfirmed =
            sourceContent?.success === true
            &&
            sourceContent?.valid === true;


        const claimsConfirmed =
            claims?.success === true
            &&
            claims?.valid === true;


        if(
            sourceConfirmed
            &&
            claimsConfirmed
        ){

            console.log(
                "Jessica Validator AI skipped: source and claims confirmed"
            );


            return buildSuccessResult(

                "Источник и утверждения ответа подтверждены",

                OUTCOME_RESULT,

                {

                    validationMode:
                        MODE_SPECIALIZED,

                    degraded:
                        false

                }

            );

        }


        console.log(
            "Jessica Validator: specialized validation unavailable, using semantic fallback"
        );


        return await runSemanticValidation(

            task,

            plan,

            taskRunResult,

            answerResult,

            {

                fallbackFromSpecialized:
                    true

            }

        );

    }


    /*
     * =====================================================
     * 5. SEARCH RESULTS / NONE
     * =====================================================
     */


    return await runSemanticValidation(

        task,

        plan,

        taskRunResult,

        answerResult,

        {

            fallbackFromSpecialized:
                false

        }

    );

}
