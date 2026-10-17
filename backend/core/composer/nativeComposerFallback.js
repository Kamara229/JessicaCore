/*
 * =========================================================
 * JESSICA NATIVE COMPOSER FALLBACK v1
 * =========================================================
 *
 * Best-effort пользовательский ответ
 * из уже полученных Tool Results.
 *
 *
 * Используется только если AI Composer
 * недоступен или исчерпал technical retry.
 *
 *
 * Принцип:
 *
 * - не придумывать факты;
 * - не решать задачу заново;
 * - показывать только данные,
 *   реально полученные Jessica;
 * - отдавать полезный частичный результат,
 *   а не превращать provider limit
 *   в полный FAILURE.
 *
 * =========================================================
 */


const MAX_ANSWER_LENGTH =
    3000;


const STOP_WORDS =
    new Set([

        "найди",
        "найти",
        "проверь",
        "определи",
        "актуальной",
        "актуальный",
        "текущей",
        "текущий",
        "стабильной",
        "стабильный",
        "версии",
        "версия",
        "дату",
        "дата",
        "выпуска",
        "официальном",
        "официальный",
        "сайте",
        "сайт",
        "затем",
        "рассчитай",
        "сколько",
        "полных",
        "дней",
        "между",

        "find",
        "check",
        "current",
        "latest",
        "stable",
        "version",
        "release",
        "date",
        "official",
        "site",
        "website",
        "then",
        "calculate",
        "days"

    ]);


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


function tokenize(
    value
) {

    return [

        ...new Set(

            normalizeText(
                value
            )
                .toLowerCase()
                .split(
                    /[^\p{L}\p{N}+#._-]+/u
                )
                .map(
                    item =>
                        item
                            .replace(
                                /^[._-]+|[._-]+$/g,
                                ""
                            )
                )
                .filter(
                    item =>
                        item.length >= 3
                        &&
                        !STOP_WORDS.has(
                            item
                        )
                )

        )

    ]
        .slice(
            0,
            24
        );

}


function successfulResults(
    taskRunResult
) {

    return Array.isArray(
        taskRunResult?.results
    )

        ? taskRunResult.results.filter(
            item =>
                item?.success === true
        )

        : [];

}


function scoreText(
    text,
    tokens
) {

    const source =
        normalizeText(
            text
        )
            .toLowerCase();


    if(!source){

        return 0;

    }


    let score = 0;


    for(
        const token
        of tokens
    ){

        if(
            source.includes(
                token
            )
        ){

            score += 1;

        }

    }


    if(
        /\b\d{4}-\d{2}-\d{2}\b/.test(
            source
        )

        ||

        /\b\d{1,2}\s+[\p{L}.]+\s+\d{4}\b/iu.test(
            source
        )
    ){

        score += 2;

    }


    if(
        /\bv?\d{1,3}(?:\.\d{1,4}){1,3}\b/i.test(
            source
        )
    ){

        score += 1;

    }


    return score;

}


function extractRelevantFragments(
    content,
    task
) {

    const tokens =

        tokenize(
            task
        );


    const lines =

        normalizeText(
            content
        )
            .split(
                /\n+|(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])/u
            )
            .map(
                line =>
                    line
                        .replace(
                            /\s+/g,
                            " "
                        )
                        .trim()
            )
            .filter(
                line =>
                    line.length >= 20
                    &&
                    line.length <= 900
            );


    const ranked =

        lines
            .map(
                (
                    line,
                    index
                ) => ({

                    line,

                    index,

                    score:
                        scoreText(
                            line,
                            tokens
                        )

                })
            )
            .filter(
                item =>
                    item.score > 0
            )
            .sort(
                (a, b) =>

                    b.score - a.score

                    ||

                    a.index - b.index
            );


    const selected = [];

    const known =
        new Set();


    for(
        const item
        of ranked
    ){

        const key =
            item.line.toLowerCase();


        if(
            known.has(
                key
            )
        ){

            continue;

        }


        known.add(
            key
        );


        selected.push(
            item.line
        );


        if(
            selected.length >= 4
        ){

            break;

        }

    }


    return selected;

}


function buildFetchFallback(
    task,
    result
) {

    const content =

        normalizeText(
            result?.data?.content
        );


    if(!content){

        return null;

    }


    const fragments =

        extractRelevantFragments(

            content,

            task

        );


    if(
        fragments.length === 0
    ){

        return null;

    }


    const title =

        normalizeText(
            result?.data?.title
        );


    const url =

        normalizeText(
            result?.data?.url
        );


    return [

        title
            ? "Источник: " + title
            : "Источник загружен.",

        url
            ? url
            : null,

        "",

        "Релевантные фрагменты источника:",

        ...fragments.map(
            item =>
                "- " + item
        )

    ]
        .filter(
            item =>
                item !== null
        )
        .join(
            "\n"
        )
        .slice(
            0,
            MAX_ANSWER_LENGTH
        );

}


function buildSearchFallback(
    results
) {

    const items =

        Array.isArray(
            results
        )

            ? results
                .slice(
                    0,
                    5
                )
                .map(
                    item => {

                        const title =
                            normalizeText(
                                item?.title
                            );


                        const url =
                            normalizeText(
                                item?.url
                            );


                        const snippet =
                            normalizeText(
                                item?.snippet
                            );


                        if(
                            !title
                            &&
                            !url
                            &&
                            !snippet
                        ){

                            return null;

                        }


                        return (

                            "- "
                            +
                            (
                                title

                                ||

                                url

                                ||

                                "Источник"
                            )
                            +
                            (
                                snippet
                                    ? ": " + snippet
                                    : ""
                            )
                            +
                            (
                                url
                                    ? " (" + url + ")"
                                    : ""
                            )

                        );

                    }
                )
                .filter(Boolean)

            : [];


    if(
        items.length === 0
    ){

        return null;

    }


    return [

        "Удалось получить следующие результаты поиска:",

        ...items

    ]
        .join(
            "\n"
        )
        .slice(
            0,
            MAX_ANSWER_LENGTH
        );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function buildNativeComposerFallback(

    task,

    taskRunResult

) {

    const results =

        successfulResults(
            taskRunResult
        );


    if(
        results.length === 0
    ){

        return null;

    }


    /*
     * Уже готовый текст Tool.
     */


    for(
        const result
        of results
    ){

        const text =

            normalizeText(
                result?.text
            );


        if(text){

            return {

                success:
                    true,

                text:
                    text.slice(
                        0,
                        MAX_ANSWER_LENGTH
                    ),

                source:
                    "native-fallback"

            };

        }

    }


    /*
     * Загруженный источник.
     */


    for(
        const result
        of [
            ...results
        ]
            .reverse()
    ){

        if(
            result?.tool !== "web_fetch"
        ){

            continue;

        }


        const text =

            buildFetchFallback(

                task,

                result

            );


        if(text){

            return {

                success:
                    true,

                text,

                source:
                    "native-fallback"

            };

        }

    }


    /*
     * Поисковая выдача.
     */


    for(
        const result
        of [
            ...results
        ]
            .reverse()
    ){

        if(
            result?.tool !== "web_search"
        ){

            continue;

        }


        const text =

            buildSearchFallback(
                result?.data?.results
            );


        if(text){

            return {

                success:
                    true,

                text,

                source:
                    "native-fallback"

            };

        }

    }


    return null;

}
