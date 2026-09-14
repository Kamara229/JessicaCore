import {
    requestLearningAnalysis
} from "./learningAnalyzerRequest.js";

import {
    parseLearningAnalysis
} from "./learningAnalyzerParser.js";

import {
    validateLearningAnalysis
} from "./learningAnalyzerValidator.js";

import {
    validateLearningGrounding
} from "./learningGroundingValidator.js";

import {
    validateLearningGeneralization
} from "./learningGeneralizationValidator.js";


/*
 * =========================================================
 * JESSICA LEARNING ANALYZER
 * =========================================================
 *
 * Центральный координатор анализа
 * исправления пользователя.
 *
 *
 * Рабочая цепочка:
 *
 * correction
 *    ↓
 * AI Request
 *    ↓
 * Parser
 *    ↓
 * Structure Validator
 *    ↓
 * Grounding Validator
 *    ↓
 * Generalization Validator
 *    ↓
 * Valid Learning Analysis
 *
 *
 * Этот модуль НЕ:
 *
 * - сохраняет Experience;
 * - создаёт новую версию Skill;
 * - подтверждает обучение;
 * - работает с Supabase;
 * - повторно выполняет задачу;
 * - изменяет существующие Skills.
 *
 * =========================================================
 */


/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function createFailureResult({

    stage,

    error,

    rawText = "",

    validationErrors = [],

    groundingErrors = [],

    generalizationErrors = []

} = {}) {


    return {

        success:
            false,

        stage:
            stage || "unknown",

        analysis:
            null,

        rawText:
            String(
                rawText || ""
            ),

        validationErrors:
            Array.isArray(
                validationErrors
            )
                ? validationErrors
                : [],

        groundingErrors:
            Array.isArray(
                groundingErrors
            )
                ? groundingErrors
                : [],

        generalizationErrors:
            Array.isArray(
                generalizationErrors
            )
                ? generalizationErrors
                : [],

        error:
            String(
                error ||
                "Неизвестная ошибка Learning Analyzer"
            )

    };

}


/*
 * =========================================================
 * ANALYZE USER CORRECTION
 * =========================================================
 */


export async function analyzeUserCorrection({

    task,

    previousAnswer = "",

    correction,

    correctedAnswer = ""

} = {}) {


    /*
     * =====================================================
     * 1. AI REQUEST
     * =====================================================
     */


    let rawText;


    try {


        rawText =
            await requestLearningAnalysis({

                task,

                previousAnswer,

                correction,

                correctedAnswer

            });


    } catch (error) {


        console.error(
            "Learning Analyzer request error:",
            error
        );


        return createFailureResult({

            stage:
                "request",

            error:
                error?.message ||
                "Не удалось выполнить Learning Analysis"

        });

    }


    /*
     * =====================================================
     * 2. PARSE
     * =====================================================
     */


    const parseResult =
        parseLearningAnalysis(
            rawText
        );


    if (
        !parseResult?.success ||
        !parseResult?.data
    ) {


        console.error(
            "Learning Analyzer parse error:",
            parseResult?.error
        );


        return createFailureResult({

            stage:
                "parse",

            error:
                parseResult?.error ||
                "Не удалось разобрать ответ Learning Analyzer",

            rawText

        });

    }


    /*
     * =====================================================
     * 3. STRUCTURE VALIDATION
     * =====================================================
     */


    const validationResult =
        validateLearningAnalysis(
            parseResult.data
        );


    if (
        !validationResult?.valid
    ) {


        console.error(
            "Learning Analyzer validation error:",
            validationResult?.errors
        );


        return createFailureResult({

            stage:
                "validation",

            error:
                "Learning Analyzer вернул неполную или некорректную структуру",

            rawText,

            validationErrors:
                validationResult?.errors || []

        });

    }


    /*
     * =====================================================
     * 4. GROUNDING VALIDATION
     * =====================================================
     *
     * Проверяем:
     *
     * - есть ли evidence для каждого правила;
     * - совпадает ли evidence
     *   с реальным correction пользователя;
     * - нет ли лишних или дублирующих evidence.
     *
     * =====================================================
     */


    const groundingResult =
        validateLearningGrounding({

            correction,

            analysis:
                parseResult.data

        });


    if (
        !groundingResult?.valid
    ) {


        console.error(
            "Learning Analyzer grounding error:",
            groundingResult?.errors
        );


        return createFailureResult({

            stage:
                "grounding",

            error:
                "Learning Analyzer предложил правила, которые не подтверждены исправлением пользователя",

            rawText,

            groundingErrors:
                groundingResult?.errors || []

        });

    }


    /*
     * =====================================================
     * 5. GENERALIZATION VALIDATION
     * =====================================================
     *
     * Grounding ещё не гарантирует,
     * что Skill является общим.
     *
     * Например:
     *
     * correction действительно содержит:
     *
     * "Для Blender правильный домен — blender.org"
     *
     * Поэтому blender.org является
     * подтверждённым фактом.
     *
     * Но он не должен попадать
     * в reusable Skill.
     *
     *
     * Здесь проверяем утечку
     * конкретных значений примера:
     *
     * - URL;
     * - домены;
     * - email;
     * - UUID;
     * - длинные идентификаторы.
     *
     * =====================================================
     */


    const generalizationResult =
        validateLearningGeneralization({

            task,

            correction,

            correctedAnswer,

            analysis:
                parseResult.data

        });


    if (
        !generalizationResult?.valid
    ) {


        console.error(
            "Learning Analyzer generalization error:",
            generalizationResult?.errors
        );


        return createFailureResult({

            stage:
                "generalization",

            error:
                "Learning Analyzer перенёс конкретные данные единичного примера в переиспользуемый Skill",

            rawText,

            generalizationErrors:
                generalizationResult?.errors || []

        });

    }


    /*
     * =====================================================
     * 6. SUCCESS
     * =====================================================
     *
     * До SUCCESS результат доходит только если:
     *
     * 1. AI вернул ответ;
     * 2. JSON разобран;
     * 3. структура валидна;
     * 4. правила подтверждены correction;
     * 5. конкретные данные примера
     *    не попали в reusable Skill.
     *
     * =====================================================
     */


    return {

        success:
            true,

        stage:
            "completed",

        analysis:
            parseResult.data,

        grounding: {

            valid:
                true,

            checkedFields:
                Array.isArray(
                    groundingResult?.checkedFields
                )
                    ? groundingResult.checkedFields
                    : [],

            checkedEvidenceCount:
                Number(
                    groundingResult
                        ?.checkedEvidenceCount || 0
                )

        },

        generalization: {

            valid:
                true,

            checked:
                generalizationResult?.checked === true,

            checkedFields:
                Array.isArray(
                    generalizationResult?.checkedFields
                )
                    ? generalizationResult.checkedFields
                    : [],

            exampleMarkers:
                Array.isArray(
                    generalizationResult?.exampleMarkers
                )
                    ? generalizationResult.exampleMarkers
                    : []

        },

        rawText:
            rawText || "",

        validationErrors:
            [],

        groundingErrors:
            [],

        generalizationErrors:
            [],

        error:
            ""

    };

}
