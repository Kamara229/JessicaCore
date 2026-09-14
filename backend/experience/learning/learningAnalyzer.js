import {
    requestLearningAnalysis
} from "./learningAnalyzerRequest.js";

import {
    parseLearningAnalysis
} from "./learningAnalyzerParser.js";

import {
    validateLearningAnalysis
} from "./learningAnalyzerValidator.js";


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
 * Validator
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
 * EMPTY FAILURE
 * =========================================================
 */


function createFailureResult({

    stage,

    error,

    rawText = "",

    validationErrors = []

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
     * 3. VALIDATE
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
     * 4. SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        stage:
            "completed",

        analysis:
            parseResult.data,

        rawText:
            rawText || "",

        validationErrors:
            [],

        error:
            ""

    };

}
