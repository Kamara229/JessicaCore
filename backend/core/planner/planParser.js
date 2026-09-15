/*
 * =========================================================
 * JESSICA PLAN PARSER
 * =========================================================
 *
 * Преобразует ответ Planner AI
 * в JavaScript object.
 *
 *
 * Ответственность:
 *
 * AI response
 *      ↓
 * JSON extraction
 *      ↓
 * JavaScript object
 *
 *
 * НЕ отвечает за:
 *
 * - нормализацию;
 * - валидацию;
 * - исправление плана.
 *
 * =========================================================
 */



/*
 * =========================================================
 * CLEAN JSON TEXT
 * =========================================================
 */


function cleanJsonText(
    text
) {


    let value =
        String(
            text || ""
        )
        .trim();



    if (!value) {

        return "";

    }



    /*
     * Убираем markdown:
     *
     * ```json
     * {}
     * ```
     */


    value =
        value.replace(
            /^```(?:json)?/i,
            ""
        );


    value =
        value.replace(
            /```$/i,
            ""
        );


    value =
        value.trim();



    /*
     * Если модель добавила текст
     * до/после JSON,
     * пытаемся найти объект.
     */


    const firstBrace =
        value.indexOf(
            "{"
        );


    if (
        firstBrace === -1
    ) {

        return value;

    }



    let depth =
        0;


    let started =
        false;



    for (
        let i = firstBrace;
        i < value.length;
        i++
    ) {


        const char =
            value[i];


        if (
            char === "{"
        ) {

            depth++;

            started =
                true;

        }


        if (
            char === "}"
        ) {

            depth--;

        }


        if (
            started &&
            depth === 0
        ) {

            return value.slice(
                firstBrace,
                i + 1
            );

        }

    }



    return value;

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



    if (
        !cleaned
    ) {

        throw new Error(
            "Planner вернул пустой ответ"
        );

    }



    let parsed;



    try {


        parsed =
            JSON.parse(
                cleaned
            );


    } catch(error) {


        console.error(
            "Jessica Planner JSON parse error:",
            cleaned
        );


        throw new Error(
            "Planner вернул невалидный JSON"
        );

    }



    /*
     * Planner должен вернуть объект,
     * а не массив или примитив.
     */


    if (
        !parsed ||
        typeof parsed !== "object" ||
        Array.isArray(
            parsed
        )
    ) {

        throw new Error(
            "Planner JSON должен быть объектом"
        );

    }



    return parsed;

}
