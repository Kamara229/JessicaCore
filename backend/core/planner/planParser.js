/*
 * =========================================================
 * JESSICA PLAN PARSER
 * =========================================================
 *
 * Отвечает только за преобразование
 * ответа Planner в JavaScript object.
 *
 * Не содержит:
 *
 * - AI запросов
 * - Planner логики
 * - нормализации плана
 * - валидации плана
 *
 * =========================================================
 */


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =
        String(
            text || ""
        ).trim();


    /*
     * Удаляем markdown code block:
     *
     * ```json
     * {...}
     * ```
     */


    if (
        value.startsWith(
            "```"
        )
    ) {

        value =
            value.replace(
                /^```(?:json)?\s*/i,
                ""
            );


        value =
            value.replace(
                /\s*```$/,
                ""
            );

    }


    /*
     * Если модель добавила текст
     * вокруг JSON, пытаемся извлечь
     * объект между первой и последней
     * фигурной скобкой.
     */


    const firstBrace =
        value.indexOf(
            "{"
        );


    const lastBrace =
        value.lastIndexOf(
            "}"
        );


    if (
        firstBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value.trim();

}


/*
 * =========================================================
 * PARSE PLAN
 * =========================================================
 */


export function parsePlan(
    text
) {

    const cleaned =
        cleanJsonText(
            text
        );


    if (!cleaned) {

        throw new Error(
            "Planner вернул пустой ответ"
        );

    }


    try {

        return JSON.parse(
            cleaned
        );

    } catch {

        throw new Error(
            "Planner вернул невалидный JSON"
        );

    }

}
