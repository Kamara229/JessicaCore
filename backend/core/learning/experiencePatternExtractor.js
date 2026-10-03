/*
 * =========================================================
 * JESSICA EXPERIENCE PATTERN EXTRACTOR v2
 * =========================================================
 *
 * Центральный координатор автономного
 * извлечения нового Experience Pattern.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Evidence Builder
 *        ↓
 * AI Request
 *        ↓
 * Parser
 *        ↓
 * Pattern Normalizer
 *        ↓
 * Pattern Validator
 *        ↓
 * Dynamic Experience Pattern
 *
 *
 * Ответственность:
 *
 * - координировать Pattern Extraction;
 * - передавать данные между слоями;
 * - возвращать унифицированный результат.
 *
 *
 * НЕ:
 *
 * - строит Prompt;
 * - вызывает AI напрямую;
 * - анализирует Trace самостоятельно;
 * - парсит AI JSON;
 * - нормализует Pattern;
 * - сохраняет Experience;
 * - принимает AUTO_APPROVE.
 *
 * =========================================================
 */


import {
    buildPatternEvidence
} from "./patternExtraction/patternEvidence.js";


import {
    requestPatternExtraction
} from "./patternExtraction/patternRequest.js";


import {
    parsePatternResponse
} from "./patternExtraction/patternParser.js";


import {
    normalizeExtractedPattern
} from "./patternExtraction/patternNormalizer.js";


import {
    validatePattern
} from "./patternExtraction/patternValidator.js";





/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function buildFailureResult({

    stage,

    reason,

    evidence = null,

    rawText = ""

} = {}) {


    return {


        success:
            false,


        reusable:
            false,


        stage:

            stage ||

            "unknown",


        reason:

            normalizeText(
                reason
            )

            ||

            "Experience Pattern extraction failed",


        pattern:
            null,


        evidence,


        rawText:

            normalizeText(
                rawText
            )

    };

}





/*
 * =========================================================
 * EXTRACT EXPERIENCE PATTERN
 * =========================================================
 */


export async function extractExperiencePattern({

    trace,

    metrics = {}

} = {}) {


    /*
     * =====================================================
     * 1. INPUT
     * =====================================================
     */


    if(
        !isObject(
            trace
        )
    ){

        return buildFailureResult({

            stage:
                "input",

            reason:
                "Execution Trace отсутствует"

        });

    }



    if(
        !normalizeText(
            trace?.task
        )
    ){

        return buildFailureResult({

            stage:
                "input",

            reason:
                "Task отсутствует"

        });

    }



    /*
     * =====================================================
     * 2. EVIDENCE
     * =====================================================
     */


    const evidence =

        buildPatternEvidence({

            trace,

            metrics

        });



    /*
     * =====================================================
     * 3. AI REQUEST
     * =====================================================
     */


    const requestResult =

        await requestPatternExtraction(
            evidence
        );



    if(
        !requestResult?.success
    ){

        return buildFailureResult({

            stage:
                "request",

            reason:

                requestResult?.error

                ||

                "Pattern Extractor request failed",

            evidence

        });

    }



    const rawText =

        requestResult.rawText ||

        "";



    /*
     * =====================================================
     * 4. PARSE
     * =====================================================
     */


    const parsed =

        parsePatternResponse(
            rawText
        );



    if(
        !parsed?.success
    ){

        return buildFailureResult({

            stage:
                "parse",

            reason:

                parsed?.error

                ||

                "Pattern Extractor вернул некорректный JSON",

            evidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 5. REUSABILITY
     * =====================================================
     */


    if(
        parsed.data?.reusable !== true
    ){

        return {


            success:
                true,


            reusable:
                false,


            stage:
                "completed",


            reason:

                normalizeText(
                    parsed.data?.reason
                )

                ||

                "Execution не содержит переиспользуемого опыта",


            pattern:
                null,


            evidence,


            rawText

        };

    }



    /*
     * =====================================================
     * 6. NORMALIZE PATTERN
     * =====================================================
     */


    const pattern =

        normalizeExtractedPattern({

            pattern:

                parsed.data?.pattern,

            evidence

        });



    if(
        !pattern
    ){

        return buildFailureResult({

            stage:
                "normalization",

            reason:
                "Не удалось нормализовать Experience Pattern",

            evidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 7. VALIDATE PATTERN
     * =====================================================
     */


    const validation =

        validatePattern(
            pattern
        );



    if(
        !validation?.valid
    ){

        return buildFailureResult({

            stage:

                validation?.stage

                ||

                "validation",

            reason:

                validation?.reason

                ||

                "Experience Pattern не прошёл проверку",

            evidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 8. SUCCESS
     * =====================================================
     */


    return {


        success:
            true,


        reusable:
            true,


        stage:
            "completed",


        reason:

            normalizeText(
                parsed.data?.reason
            )

            ||

            "Обнаружен новый переиспользуемый Experience Pattern",


        pattern,


        evidence,


        validation,


        rawText

    };

}
