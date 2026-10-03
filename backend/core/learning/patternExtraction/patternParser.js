/*
 * =========================================================
 * JESSICA PATTERN PARSER v1
 * =========================================================
 *
 * Парсинг JSON ответа
 * Experience Pattern Extractor.
 *
 * =========================================================
 */





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





function extractJsonText(
    rawText
) {

    let text =

        normalizeText(
            rawText
        );



    if(
        text.startsWith(
            "```"
        )
    ){

        text = text

            .replace(
                /^```(?:json)?\s*/i,
                ""
            )

            .replace(
                /\s*```$/,
                ""
            );

    }



    const firstBrace =

        text.indexOf(
            "{"
        );



    const lastBrace =

        text.lastIndexOf(
            "}"
        );



    if(
        firstBrace >= 0
        &&
        lastBrace > firstBrace
    ){

        return text.slice(
            firstBrace,
            lastBrace + 1
        );

    }



    return text;

}





export function parsePatternResponse(
    rawText
) {

    const jsonText =

        extractJsonText(
            rawText
        );



    if(
        !jsonText
    ){

        return {


            success:
                false,


            data:
                null,


            error:
                "Pattern Extractor вернул пустой JSON"

        };

    }



    try {


        const data =

            JSON.parse(
                jsonText
            );



        return {


            success:
                true,


            data,


            error:
                null

        };


    }catch(error){


        return {


            success:
                false,


            data:
                null,


            error:

                error?.message

                ||

                "Invalid Pattern JSON"

        };

    }

}
