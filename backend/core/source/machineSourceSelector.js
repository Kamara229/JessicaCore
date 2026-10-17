/*
 * =========================================================
 * JESSICA MACHINE SOURCE SELECTOR v1
 * =========================================================
 *
 * Детерминированный выбор кандидата
 * из web_search без AI.
 *
 *
 * Используется:
 *
 * 1. первым — когда machine confidence высокий;
 * 2. как fallback — когда AI Source Selector недоступен.
 *
 *
 * НЕ утверждает, что страница содержит факт.
 * Окончательная проверка выполняется web_fetch + Validator.
 *
 * =========================================================
 */


const STOP_WORDS =
    new Set([

        "найди",
        "найти",
        "проверь",
        "проверить",
        "определи",
        "определить",
        "актуальной",
        "актуальная",
        "актуальный",
        "текущей",
        "текущая",
        "текущий",
        "стабильной",
        "стабильная",
        "стабильный",
        "версии",
        "версия",
        "дату",
        "дата",
        "выпуска",
        "релиза",
        "релиз",
        "официальном",
        "официальный",
        "официального",
        "сайте",
        "сайт",
        "источник",
        "источнике",
        "затем",
        "рассчитай",
        "сколько",
        "полных",
        "дней",
        "прошло",
        "между",
        "используй",
        "найденную",
        "первой",
        "части",
        "обязательно",

        "find",
        "check",
        "verify",
        "current",
        "latest",
        "stable",
        "version",
        "release",
        "date",
        "official",
        "website",
        "site",
        "source",
        "then",
        "calculate",
        "days",
        "between",
        "using"

    ]);


const AGGREGATOR_HOSTS = [

    "wikipedia.org",
    "reddit.com",
    "stackoverflow.com",
    "stackexchange.com",
    "endoflife.date",
    "medium.com",
    "dev.to",
    "quora.com"

];


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


function tokenize(
    value
) {

    const tokens =

        normalizeText(
            value
        )
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
                    &&
                    !/^\d{4}$/.test(
                        item
                    )
            );


    return [
        ...new Set(tokens)
    ]
        .slice(
            0,
            24
        );

}


function getHostname(
    url
) {

    try {

        return new URL(
            url
        )
            .hostname
            .toLowerCase()
            .replace(
                /^www\./,
                ""
            );

    }catch(error){

        return "";

    }

}


function hostMatches(
    hostname,
    domain
) {

    return (

        hostname === domain

        ||

        hostname.endsWith(
            "." + domain
        )

    );

}


function isAggregator(
    hostname
) {

    return AGGREGATOR_HOSTS.some(
        domain =>
            hostMatches(
                hostname,
                domain
            )
    );

}


function requiresOfficialSource(
    task
) {

    return (

        /официальн\p{L}*/iu.test(
            task
        )

        ||

        /\bofficial\b/iu.test(
            task
        )

    );

}


function countMatches(
    tokens,
    text
) {

    const source =
        normalizeText(
            text
        );


    let count = 0;


    for(
        const token
        of tokens
    ){

        if(
            source.includes(
                token
            )
        ){

            count += 1;

        }

    }


    return count;

}


function scoreCandidate({

    taskTokens,

    officialRequired,

    candidate

}) {

    const hostname =

        getHostname(
            candidate?.url
        );


    const titleMatches =

        countMatches(

            taskTokens,

            candidate?.title

        );


    const snippetMatches =

        countMatches(

            taskTokens,

            candidate?.snippet

        );


    const urlMatches =

        countMatches(

            taskTokens,

            hostname
            +
            " "
            +
            candidate?.url

        );


    let score =

        titleMatches * 3

        +

        snippetMatches

        +

        urlMatches * 5;


    const hostTokenMatch =

        taskTokens.some(

            token =>

                token.length >= 4

                &&

                hostname.includes(
                    token
                )

        );


    if(hostTokenMatch){

        score += 8;

    }


    if(
        officialRequired
    ){

        if(hostTokenMatch){

            score += 10;

        }


        if(
            /\bofficial\b/i.test(
                candidate?.title || ""
            )

            ||

            /официальн\p{L}*/iu.test(
                candidate?.title || ""
            )
        ){

            score += 2;

        }


        if(
            isAggregator(
                hostname
            )
        ){

            score -= 18;

        }

    }


    if(
        /\/(docs?|documentation|release|releases|download|news)(\/|$)/i.test(
            candidate?.url || ""
        )
    ){

        score += 2;

    }


    return {

        score,

        hostname,

        hostTokenMatch,

        titleMatches,

        snippetMatches,

        urlMatches

    };

}


export function selectSourceMachine(

    task,

    results,

    {
        allowWeak =
            false
    } = {}

) {

    const candidates =

        Array.isArray(
            results
        )

            ? results

            : [];


    if(
        candidates.length === 0
    ){

        return {

            selected:
                false,

            result:
                null,

            index:
                null,

            score:
                0,

            reason:
                "Нет кандидатов"

        };

    }


    const taskTokens =

        tokenize(
            task
        );


    const officialRequired =

        requiresOfficialSource(
            task
        );


    const ranked =

        candidates
            .map(
                (
                    candidate,
                    position
                ) => ({

                    candidate,

                    position,

                    details:

                        scoreCandidate({

                            taskTokens,

                            officialRequired,

                            candidate

                        })

                })
            )
            .sort(
                (a, b) =>
                    b.details.score
                    -
                    a.details.score
            );


    const top =
        ranked[0];


    const second =
        ranked[1];


    if(!top){

        return {

            selected:
                false,

            result:
                null,

            index:
                null,

            score:
                0,

            reason:
                "Нет ранжированных кандидатов"

        };

    }


    const score =
        top.details.score;


    const margin =

        second

            ? score - second.details.score

            : score;


    const strongOfficialMatch =

        officialRequired

        &&

        top.details.hostTokenMatch

        &&

        !isAggregator(
            top.details.hostname
        );


    const strong =

        strongOfficialMatch

        ||

        (
            score >= 12
            &&
            margin >= 3
        );


    const weak =

        score >= 4;


    if(
        !strong

        &&

        !(
            allowWeak
            &&
            weak
        )
    ){

        return {

            selected:
                false,

            result:
                null,

            index:
                null,

            score,

            margin,

            reason:
                "Machine confidence недостаточен"

        };

    }


    return {

        selected:
            true,

        index:

            Number.isInteger(
                top.candidate?.index
            )

                ? top.candidate.index

                : top.position,

        result:
            top.candidate,

        score,

        margin,

        officialRequired,

        strong,

        reason:

            strong

                ? "Источник выбран детерминированно с высокой уверенностью"

                : "Источник выбран как best-effort fallback без AI"

    };

}
