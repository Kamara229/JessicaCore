/*
 * =========================================================
 * JESSICA EXPERIENCE PATTERN EXTRACTOR v3
 * =========================================================
 *
 * Центральный координатор автономного
 * извлечения нового Experience Pattern.
 *
 *
 * Flow:
 *
 * Trace / Prepared Evidence
 *        ↓
 * Evidence
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
 * - принимать Trace или готовый Evidence;
 * - передавать данные между слоями;
 * - возвращать унифицированный результат.
 *
 *
 * НЕ:
 *
 * - строит Prompt;
 * - вызывает AI напрямую;
 * - парсит JSON;
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
 * RESOLVE EVIDENCE
 * =========================================================
 */


function resolveEvidence({

    trace,

    evidence,

    metrics

}) {


    if(
        isObject(
            evidence
        )
    ){

        return evidence;

    }



    if(
        !isObject(
            trace
        )
    ){

        return null;

    }



    return buildPatternEvidence({

        trace,

        metrics

    });

}





/*
 * =========================================================
 * EXTRACT EXPERIENCE PATTERN
 * =========================================================
 */


export async function extractExperiencePattern({

    trace = null,

    evidence = null,

    metrics = {}

} = {}) {


    /*
     * =====================================================
     * 1. EVIDENCE
     * =====================================================
     */


    const resolvedEvidence =

        resolveEvidence({

            trace,

            evidence,

            metrics

        });



    if(
        !isObject(
            resolvedEvidence
        )
    ){

        return buildFailureResult({

            stage:
                "input",

            reason:
                "Learning Evidence отсутствует"

        });

    }



    if(
        !normalizeText(
            resolvedEvidence.task
        )
    ){

        return buildFailureResult({

            stage:
                "input",

            reason:
                "Task отсутствует в Learning Evidence",

            evidence:
                resolvedEvidence

        });

    }



    /*
     * =====================================================
     * 2. AI REQUEST
     * =====================================================
     */


    const requestResult =

        await requestPatternExtraction(
            resolvedEvidence
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

            evidence:
                resolvedEvidence

        });

    }



    const rawText =

        requestResult.rawText ||

        "";



    /*
     * =====================================================
     * 3. PARSE
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

            evidence:
                resolvedEvidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 4. REUSABILITY
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


            evidence:
                resolvedEvidence,


            rawText

        };

    }



    /*
     * =====================================================
     * 5. NORMALIZE
     * =====================================================
     */


    const pattern =

        normalizeExtractedPattern({

            pattern:

                parsed.data?.pattern,

            evidence:

                resolvedEvidence

        });



    if(
        !pattern
    ){

        return buildFailureResult({

            stage:
                "normalization",

            reason:
                "Не удалось нормализовать Experience Pattern",

            evidence:
                resolvedEvidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 6. VALIDATION
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

            evidence:
                resolvedEvidence,

            rawText

        });

    }



    /*
     * =====================================================
     * 7. SUCCESS
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


        evidence:
            resolvedEvidence,


        validation,


        rawText

    };

}
