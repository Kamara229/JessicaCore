/*
 * =========================================================
 * JESSICA EXPERIENCE TEXT v0.5
 * =========================================================
 *
 * Semantic normalization layer.
 *
 * НЕ:
 *
 * - считает score;
 * - выбирает Skill;
 * - работает с Storage.
 *
 * =========================================================
 */



export function normalizeExperienceText(
    value
){

    return String(
        value || ""
    )
    .toLowerCase()

    .replace(
        /[_-]+/g,
        " "
    )

    .replace(
        /[^a-zа-яё0-9\s]/gi,
        " "
    )

    .replace(
        /\s+/g,
        " "
    )

    .trim();

}






export function tokenizeExperienceText(
    value
){

    const text =
        normalizeExperienceText(
            value
        );


    if(
        !text
    ){

        return [];

    }


    return text
        .split(" ")
        .filter(
            token =>
                token.length >= 2
        );

}








const CONCEPTS = [
    // Bilingual task vocabulary for reusable, domain-sensitive Skill matching.
    {"concept":"compare","patterns":["compare","comparison","comparative","contrast","contrasting","сравн","сопостав","различ","отлич"]},
    {"concept":"patent","patterns":["patent","патент"]},
    {"concept":"license","patterns":["license","licence","лиценз"]},
    {"concept":"clause","patterns":["clause","provision","section","пункт","раздел","положен","услов","оговорк"]},
    {"concept":"terminate","patterns":["terminat","termination","прекращ","расторж"]},
    {"concept":"legal","patterns":["legal","юрид","правов"]},
    {"concept":"rights","patterns":["right","право","права"]},
    {"concept":"grant","patterns":["grant","предоставл","выдач"]},
    {"concept":"source","patterns":["source","источн","ссылк"]},
    {"concept":"text","patterns":["text","текст"]},
    {"concept":"software","patterns":["software","программно"]},


    {
        concept:"verify",
        patterns:[
            "verify",
            "verified",
            "verification",
            "провер",
            "вериф"
        ]
    },


    {
        concept:"search",
        patterns:[
            "search",
            "find",
            "ищ",
            "поиск",
            "найд"
        ]
    },


    {
        concept:"official",
        patterns:[
            "official",
            "официал"
        ]
    },


    {
        concept:"website",
        patterns:[
            "website",
            "site",
            "web",
            "сайт"
        ]
    },


    {
        concept:"document",
        patterns:[
            "document",
            "doc",
            "документ"
        ]
    },


    {
        concept:"certificate",
        patterns:[
            "certificate",
            "сертификат",
            "удостовер"
        ]
    },


    {
        concept:"create",
        patterns:[
            "create",
            "make",
            "созд",
            "формир"
        ]
    },


    {
        concept:"get",
        patterns:[
            "get",
            "receive",
            "получ",
            "загруз"
        ]
    },


    {
        concept:"send",
        patterns:[
            "send",
            "отправ",
            "перед"
        ]
    },


    {
        concept:"analyze",
        patterns:[
            "analyze",
            "analysis",
            "анализ"
        ]
    },


    {
        concept:"save",
        patterns:[
            "save",
            "store",
            "сохран"
        ]
    },


    {
        concept:"api",
        patterns:[
            "api"
        ]
    },


    {
        concept:"database",
        patterns:[
            "database",
            "db",
            "база",
            "данных"
        ]
    },


    {
        concept:"code",
        patterns:[
            "code",
            "код",
            "программ"
        ]
    },


    {
        concept:"error",
        patterns:[
            "error",
            "bug",
            "ошиб"
        ]
    },


    {
        concept:"report",
        patterns:[
            "report",
            "отчет",
            "доклад"
        ]
    }


];








export function canonicalizeExperienceToken(
    token
){

    const value =
        normalizeExperienceText(
            token
        );


    if(
        !value
    ){

        return "";

    }



    for(
        const item of CONCEPTS
    ){

        for(
            const pattern of item.patterns
        ){

            if(
                value === pattern
                ||
                value.startsWith(pattern)
            ){

                return item.concept;

            }

        }

    }



    return value;

}









export function canonicalizeExperienceTokens(
    value
){

    return tokenizeExperienceText(
        value
    )

    .map(
        canonicalizeExperienceToken
    )

    .filter(Boolean);

}








export function canonicalizeExperiencePhrase(
    value
){

    return canonicalizeExperienceTokens(
        value
    );

}








export function uniqueExperienceValues(
    values
){

    if(
        !Array.isArray(values)
    ){

        return [];

    }


    return [

        ...new Set(
            values.filter(Boolean)
        )

    ];

}








export function normalizeExperienceStringArray(
    value
){

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value

        .map(
            item =>
                String(item || "")
                .trim()
        )

        .filter(Boolean);

}
