/*
 * =========================================================
 * JESSICA RESULT VALIDATOR v2
 * =========================================================
 *
 * Главный координатор проверки результата.
 *
 *
 * Flow:
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
 * При технической недоступности
 * specialized validator:
 *
 * Semantic AI Fallback
 *
 *
 * Дополнительно результат содержит:
 *
 * validationMode
 * degraded
 *
 *
 * Это позволяет Learning отличать:
 *
 * fully verified execution
 *
 * от:
 *
 * technically degraded validation path.
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
 * VALIDATION MODES
 * =========================================================
 */


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


/*
 * =========================================================
 * NORMALIZE OUTCOME TYPE
 * =========================================================
 */


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


/*
 * =========================================================
 * RESULT BUILDER
 * =========================================================
 */


function buildValidationResult(

    validation,

    {

        validationMode =
            MODE_BASIC,

        degraded =
            false

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


/*
 * =========================================================
 * SUCCESS RESULT
 * =========================================================
 */


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


    if(
        mode === "source_content"
        ||
        mode === "search_results"
    ){

        return mode;

    }


    return "none";

}


/*
 * =========================================================
 * AI SEMANTIC VALIDATION
 * =========================================================
 */


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


    /*
     * =====================================================
     * AI SUCCESS
     * =====================================================
     */


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

                    fallbackFromSpecialized

            }

        );

    }


    /*
     * =====================================================
     * AI UNAVAILABLE
     * =====================================================
     *
     * Basic + Evidence уже пройдены.
     *
     * Задачу не ломаем технической
     * недоступностью Validator AI.
     *
     * Но явно маркируем результат
     * как degraded.
     * =====================================================
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


    if(
        basic.valid !== true
    ){

        return buildValidationResult(

            basic,

            {

                validationMode:
                    MODE_BASIC,

                degraded:
                    false

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
                    false

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
                        false

                }

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
                        false

                }

            );

        }


        /*
         * -----------------------------------------------
         * SPECIALIZED SUCCESS
         * -----------------------------------------------
         */


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


        /*
         * -----------------------------------------------
         * SPECIALIZED UNAVAILABLE
         * -----------------------------------------------
         */


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
