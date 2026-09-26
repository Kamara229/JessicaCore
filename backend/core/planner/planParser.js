/*
 * =========================================================
 * JESSICA PLAN PARSER v3
 * =========================================================
 *
 * Преобразует ответ Planner AI
 * в JavaScript object.
 *
 *
 * Flow:
 *
 * AI Response
 *      ↓
 * JSON Extraction
 *      ↓
 * JSON Parse
 *      ↓
 * Planner Normalizer
 *
 *
 * НЕ:
 *
 * - нормализует поля;
 * - валидирует план;
 * - исправляет ошибки;
 * - проверяет tools.
 *
 * =========================================================
 */





const MAX_RESPONSE_LENGTH =
    30000;








/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value || ""
    )
    .trim();

}









/*
 * =========================================================
 * REMOVE MARKDOWN
 * =========================================================
 */


function removeMarkdown(
    text
) {


    let result =
        safeString(
            text
        );



    if (!result) {

        return "";

    }




    /*
     * Убираем:
     *
     * ```json
     *
     */


    result =
        result.replace(
            /^```json\s*/i,
            ""
        );




    /*
     * Убираем:
     *
     * ```
     *
     */


    result =
        result.replace(
            /```$/i,
            ""
        );



    return result.trim();

}









/*
 * =========================================================
 * EXTRACT JSON OBJECT
 * =========================================================
 *
 * Ищет первый полный JSON объект.
 *
 * Поддерживает вложенность:
 *
 * {
 *    a:{
 *       b:1
 *    }
 * }
 *
 * =========================================================
 */


function extractJsonObject(
    text
) {


    const start =
        text.indexOf(
            "{"
        );



    if (
        start === -1
    ) {

        return null;

    }



    let depth =
        0;


    let inString =
        false;


    let escaped =
        false;






    for (
        let i = start;
        i < text.length;
        i++
    ) {


        const char =
            text[i];





        /*
         * JSON string handling
         */


        if (
            char === "\\" &&
            !escaped
        ) {

            escaped =
                true;

            continue;

        }




        if (
            char === '"' &&
            !escaped
        ) {

            inString =
                !inString;

        }




        escaped =
            false;






        if (
            inString
        ) {

            continue;

        }





        if (
            char === "{"
        ) {

            depth++;

        }





        if (
            char === "}"
        ) {

            depth--;


            if (
                depth === 0
            ) {


                return text.slice(

                    start,

                    i + 1

                );

            }

        }


    }




    return null;


}









/*
 * =========================================================
 * PARSE JSON
 * =========================================================
 */


function parseJson(
    text
) {


    try {


        return JSON.parse(
            text
        );


    } catch(error) {


        return null;

    }


}









/*
 * =========================================================
 * VALID OBJECT
 * =========================================================
 */


function isValidObject(
    value
) {


    return (

        value !== null &&

        typeof value === "object" &&

        !Array.isArray(value)

    );


}









/*
 * =========================================================
 * PARSE PLAN
 * =========================================================
 */


export function parsePlan(
    text
) {


    const raw =
        safeString(
            text
        );



    if (!raw) {

        return null;

    }






    /*
     * Защита от огромного ответа AI
     */


    if (
        raw.length >
        MAX_RESPONSE_LENGTH
    ) {

        console.warn(

            "Jessica Planner response too large"

        );


        return null;

    }






    /*
     * 1. Убираем markdown
     */


    const cleaned =
        removeMarkdown(
            raw
        );






    /*
     * 2. Прямая попытка JSON
     */


    let parsed =
        parseJson(
            cleaned
        );



    if (
        isValidObject(
            parsed
        )
    ) {

        return parsed;

    }






    /*
     * 3. Извлечение JSON
     */


    const extracted =
        extractJsonObject(
            cleaned
        );



    if (!extracted) {

        return null;

    }





    parsed =
        parseJson(
            extracted
        );





    if (
        !isValidObject(
            parsed
        )
    ) {


        console.warn(

            "Jessica Planner invalid JSON:",

            extracted

        );


        return null;

    }





    return parsed;


}









/*
 * =========================================================
 * DEBUG INFO
 * =========================================================
 */


export function getParserDebugInfo(
    text
) {


    const value =
        safeString(
            text
        );



    return {


        length:
            value.length,



        hasJsonStart:
            value.includes(
                "{"
            ),



        hasJsonEnd:
            value.includes(
                "}"
            )

    };


}
