/*
 * =========================================================
 * JESSICA LEARNING ANALYZER PARSER
 * =========================================================
 *
 * Преобразует сырой ответ AI
 * в JavaScript-объект.
 *
 *
 * Этот модуль:
 *
 * - очищает Markdown code fences;
 * - находит JSON;
 * - выполняет JSON.parse();
 * - возвращает понятный результат.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - проверяет качество обучения;
 * - создаёт Learning Proposal;
 * - сохраняет Experience;
 * - подтверждает Skill;
 * - изменяет Supabase.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CLEAN RAW TEXT
 * =========================================================
 */


function cleanJsonText(
    value
) {


    let text =
        String(
            value || ""
        ).trim();


    if (!text) {

        return "";

    }


    /*
     * Удаляем возможные Markdown fences:
     *
     * ```json
     * {...}
     * ```
     *
     * или:
     *
     * ```
     * {...}
     * ```
     */


    text =
        text
            .replace(
                /^```(?:json)?\s*/i,
                ""
            )
            .replace(
                /\s*```$/,
                ""
            )
            .trim();


    /*
     * Если модель всё-таки добавила
     * текст до или после JSON,
     * пытаемся извлечь внешний объект.
     */


    const firstBrace =
        text.indexOf(
            "{"
        );


    const lastBrace =
        text.lastIndexOf(
            "}"
        );


    if (
        firstBrace !== -1 &&
        lastBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        text =
            text
                .slice(
                    firstBrace,
                    lastBrace + 1
                )
                .trim();

    }


    return text;

}


/*
 * =========================================================
 * PARSE LEARNING ANALYSIS
 * =========================================================
 */


export function parseLearningAnalysis(
    rawText
) {


    /*
     * =====================================================
     * EMPTY RESPONSE
     * =====================================================
     */


    const cleanedText =
        cleanJsonText(
            rawText
        );


    if (!cleanedText) {

        return {

            success:
                false,

            data:
                null,

            error:
                "Learning Analyzer вернул пустой ответ"

        };

    }


    /*
     * =====================================================
     * JSON PARSE
     * =====================================================
     */


    let parsed;


    try {


        parsed =
            JSON.parse(
                cleanedText
            );


    } catch (error) {


        return {

            success:
                false,

            data:
                null,

            error:
                `Некорректный JSON Learning Analyzer: ${error.message}`

        };

    }


    /*
     * =====================================================
     * ROOT OBJECT CHECK
     * =====================================================
     */


    if (
        !parsed ||
        typeof parsed !== "object" ||
        Array.isArray(
            parsed
        )
    ) {

        return {

            success:
                false,

            data:
                null,

            error:
                "Learning Analyzer должен вернуть JSON-объект"

        };

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        data:
            parsed,

        error:
            ""

    };

}
