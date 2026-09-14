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
 * Valid Learning Analysis
 *
 *
 * Grounding Validator программно проверяет,
 * что операционные правила будущего Skill
 * действительно имеют основание
 * в correction пользователя.
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

    groundingErrors = []

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
     *
     * Проверяем:
     *
     * - analysis;
     * - understanding;
     * - reusable;
     * - clarificationQuestions;
     * - unsupportedSuggestions;
     * - proposedExperience;
     * - обязательные поля Skill.
     *
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
     * Теперь проверяем не просто формат,
     * а происхождение правил.
     *
     *
     * Для каждого элемента:
     *
     * - strategy;
     * - sourcePriority;
     * - validationRules;
     * - failurePatterns;
     *
     * должно существовать groundingEvidence.
     *
     *
     * Сам evidence должен реально
     * содержаться в исходном correction
     * пользователя.
     *
     *
     * Если AI придумал правило
     * или выдумал цитату,
     * Learning блокируется здесь.
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
     * 5. SUCCESS
     * =====================================================
     *
     * До этой точки Learning Analysis
     * доходит только если:
     *
     * 1. AI вернул ответ;
     * 2. JSON успешно разобран;
     * 3. структура корректна;
     * 4. grounding подтверждён кодом.
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

        rawText:
            rawText || "",

        validationErrors:
            [],

        groundingErrors:
            [],

        error:
            ""

    };

}
