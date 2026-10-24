import { getDocumentReceipt } from "./composer/documentReceipt.js";
import { requiresVerifiedComparison } from "./composer/comparisonEvidencePolicy.js";
import {groundedPatentComparison} from "./composer/nativePatentComparison.js";
import {checkProgrammingAnswerSafety} from "./validator/programmingAnswerSafety.js";
/*
 * =========================================================
 * JESSICA RESULT VALIDATOR v4
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

                : "",

        // Preserve evidence-verifier claim details for safe partial rendering
        // after a failed semantic repair. Do not treat these as verified facts.
        claims:
            Array.isArray(validation?.claims) ? validation.claims : []

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


function getDeterministicEvidenceProof(

    taskRunResult,

    answerResult

) {

    const results =

        Array.isArray(
            taskRunResult?.results
        )

            ? taskRunResult.results

            : [];


    const result =

        [
            ...results
        ]
            .reverse()
            .find(
                item =>
                    item?.tool === "release_fact"
                    &&
                    item?.success === true
                    &&
                    item?.data?.evidenceVerified === true
            );


    if(!result){

        return null;

    }


    const version =

        String(
            result?.data?.version || ""
        )
            .trim();


    const releaseDate =

        String(
            result?.data?.releaseDate || ""
        )
            .trim();


    const answerText =

        String(
            answerResult?.text || ""
        );


    if(
        !version
        ||
        !releaseDate
        ||
        !answerText.includes(
            version
        )
        ||
        !answerText.includes(
            releaseDate
        )
    ){

        return null;

    }


    return {

        tool:
            "release_fact",

        version,

        releaseDate,

        sourceUrl:

            String(
                result?.data?.sourceUrl || ""
            )
                .trim(),

        extractor:

            String(
                result?.data?.extractor || ""
            )
                .trim()

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


    // Comparison claims require real semantic verification. A missing
    // validator is not a successful or independently verified result.
    if (requiresVerifiedComparison(task, plan)) {
        return buildValidationResult(
            {valid:false,shouldRetry:false,
             reason:"Сравнение не подтверждено: AI Validator недоступен"},
            {validationMode:MODE_UNAVAILABLE_FALLBACK,
             degraded:true,failureScope:FAILURE_SCOPE.PRESENTATION}
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


    // Static language-specific invariants are checked BEFORE AI Validator,
    // which can otherwise accept a plausible-looking yet incorrect program.
    // Failure belongs to presentation and never certifies runtime execution.
    const programmingSafety=checkProgrammingAnswerSafety(task,answerResult?.text);
    if(programmingSafety) {
        console.warn("Jessica programming answer safety rejected:",programmingSafety.reason);
        return buildValidationResult(programmingSafety,{
            validationMode:"programming-static-guard",
            degraded:false,
            failureScope:FAILURE_SCOPE.PRESENTATION
        });
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


    // Only verify source capture itself; never infer the content's
    // conclusions without the later comparison and AI validation.
    if (answerResult?.source === "document-receipt") {
        const expected = getDocumentReceipt(task, plan, taskRunResult);
        if (expected && answerResult?.url === expected.url
            && answerResult?.length === expected.length
            && answerResult?.text === expected.text) {
            return buildSuccessResult(
                "Исходный документ получен и проверен детерминированно",
                OUTCOME_RESULT,
                {validationMode:"document-receipt",degraded:false}
            );
        }
        return buildValidationResult(
            {valid:false,reason:"Документ или URL не подтверждены"},
            {validationMode:MODE_BASIC,degraded:false,failureScope:FAILURE_SCOPE.EVIDENCE}
        );
    }

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
     * 4. DETERMINISTIC SOURCE PROOF
     * =====================================================
     *
     * A deterministic extractor may already have:
     *
     * - read fetched source content;
     * - extracted structured values;
     * - verified that those values coexist in one
     *   source fragment;
     * - produced the exact user-facing answer.
     *
     * In that case an additional AI Validator call
     * adds tokens but no new evidence.
     * =====================================================
     */


    if(
        answerResult?.source === "tool"
        &&
        evidenceMode === "source_content"
    ){

        const proof =

            getDeterministicEvidenceProof(

                taskRunResult,

                answerResult

            );


        if(proof){

            console.log(

                "Jessica Validator deterministic source proof:",

                JSON.stringify(
                    proof
                )

            );


            return buildSuccessResult(

                "Источник подтверждён детерминированным extractor",

                OUTCOME_RESULT,

                {

                    validationMode:
                        MODE_TOOL_DIRECT,

                    degraded:
                        false

                }

            );

        }

    }


    /*
     * =====================================================
     * 5. SOURCE CONTENT VALIDATION
     * =====================================================
     */


    if(
        evidenceMode === "source_content"
    ){

        if (answerResult?.source === "bounded-mit-patent-question") {
            const proof=groundedPatentComparison(task,plan,taskRunResult);
            return proof?.text === answerResult.text && proof?.source === answerResult.source
                ? buildSuccessResult("MIT source text independently reconstructed",OUTCOME_RESULT,
                    {validationMode:"bounded-primary-text",degraded:false})
                : buildValidationResult({valid:false,reason:"Evidence mismatch"},
                    {validationMode:"bounded-primary-text",failureScope:FAILURE_SCOPE.PRESENTATION});
        }

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


        // A source-bounded answer is accepted only when the independently
        // reconstructed text matches the two newly verified full documents.
        if (answerResult?.source === "bounded-patent-comparison" &&
            sourceContent?.success === true && sourceContent?.valid === true) {
            const independentlyChecked = groundedPatentComparison(task, plan, taskRunResult);
            if (independentlyChecked?.text === answerResult.text) {
                return buildSuccessResult("Патентные положения проверены по двум первоисточникам",
                    OUTCOME_RESULT, {validationMode:"bounded-primary-text",degraded:false});
            }
            return buildValidationResult(
                {valid:false,reason:"Сопоставление не соответствует проверенным текстам"},
                {validationMode:"bounded-primary-text",failureScope:FAILURE_SCOPE.PRESENTATION});
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


        const sourceConfirmed =
            sourceContent?.success === true
            &&
            sourceContent?.valid === true;


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

                    /*
                     * Если Source Content уже подтверждён
                     * как достаточный для задачи, а claims
                     * итогового ответа не подтверждаются,
                     * проблема принадлежит presentation:
                     * Execution / evidence повторять не нужно.
                     *
                     * Если сам source не подтверждён,
                     * сохраняем evidence scope.
                     */

                    failureScope:

                        sourceConfirmed

                            ? FAILURE_SCOPE.PRESENTATION

                            : FAILURE_SCOPE.EVIDENCE

                }

            );

        }


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
     * 6. SEARCH RESULTS / NONE
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
