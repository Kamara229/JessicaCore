/*
 * =========================================================
 * JESSICA PLANNER GROUNDING POLICY v2
 * =========================================================
 *
 * Машинная защита Planner от:
 *
 * - повторения уже проваленных search queries;
 * - повторного использования запрещённых static URLs;
 * - выдумывания конкретных значений в tool arguments.
 *
 *
 * ВАЖНО:
 *
 * FAILED / RETRY context
 * НЕ является подтверждённым источником факта.
 *
 * Experience
 * НЕ является подтверждённым источником нового факта.
 *
 *
 * Grounded values могут приходить только из:
 *
 * - исходной задачи пользователя;
 * - validatedFacts;
 * - dependencyContext;
 * - groundedFacts.
 *
 * =========================================================
 */


/*
 * =========================================================
 * SAFE
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value ?? ""
    )
        .trim();

}


function safeArray(
    value
) {

    return Array.isArray(
        value
    )
        ? value
        : [];

}


/*
 * =========================================================
 * NORMALIZE SEARCH QUERY
 * =========================================================
 */


function normalizeQuery(
    value
) {

    return safeString(
        value
    )
        .replace(
            /\s+/g,
            " "
        )
        .toLowerCase();

}


/*
 * =========================================================
 * NORMALIZE URL
 * =========================================================
 */


function normalizeUrl(
    value
) {

    const text =
        safeString(
            value
        );


    if(
        !text
    ){

        return "";

    }


    try {


        const url =
            new URL(
                text
            );


        url.hash =
            "";


        url.protocol =
            url.protocol.toLowerCase();


        url.hostname =
            url.hostname.toLowerCase();


        if(
            url.pathname.length > 1
            &&
            url.pathname.endsWith("/")
        ){

            url.pathname =
                url.pathname.slice(
                    0,
                    -1
                );

        }


        return url.toString();


    }catch(error){


        return text;

    }

}


/*
 * =========================================================
 * UNIQUE
 * =========================================================
 */


function uniqueStrings(
    values
) {

    return [

        ...new Set(

            safeArray(
                values
            )

                .map(
                    safeString
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * TRUSTED TEXT
 * =========================================================
 */


function stringifyTrustedValue(
    value
) {

    if(
        value === null
        ||
        value === undefined
    ){

        return "";

    }


    if(
        typeof value === "string"
        ||
        typeof value === "number"
        ||
        typeof value === "boolean"
    ){

        return String(
            value
        );

    }


    if(
        Array.isArray(
            value
        )
    ){

        return value

            .map(
                stringifyTrustedValue
            )

            .filter(Boolean)

            .join("\n");

    }


    if(
        typeof value === "object"
    ){

        return Object.values(
            value
        )

            .map(
                stringifyTrustedValue
            )

            .filter(Boolean)

            .join("\n");

    }


    return "";

}


/*
 * =========================================================
 * BUILD GROUNDED TEXT
 * =========================================================
 *
 * Намеренно НЕ включаем:
 *
 * - retry reason;
 * - previousAttempt;
 * - Experience;
 * - instructions;
 * - failed tool results.
 *
 * =========================================================
 */


function buildGroundedText(
    task,
    context
) {

    const trustedParts = [

        task,

        context?.validatedFacts,

        context?.dependencyContext,

        context?.groundedFacts,

        context?.metadata?.validatedFacts,

        context?.metadata?.dependencyContext,

        context?.metadata?.groundedFacts

    ];


    return trustedParts

        .map(
            stringifyTrustedValue
        )

        .filter(Boolean)

        .join("\n")

        .toLowerCase();

}


/*
 * =========================================================
 * CONCRETE VALUES
 * =========================================================
 *
 * Здесь ловим значения,
 * которые Planner особенно опасно
 * придумывает самостоятельно:
 *
 * 16.3
 * v2.4.1
 * PostgreSQL 16 release
 * version 17
 * 2026-10-01
 * 01.10.2026
 * длинные ID
 *
 * =========================================================
 */


function extractContextualMajorVersions(
    source
) {

    const values =
        new Set();


    const patterns = [

        /\bv(\d{1,4})\b/gi,

        /\b(?:version|release|major|series|branch)\s+v?(\d{1,4})\b/gi,

        /(?:верси\p{L}*|релиз\p{L}*)\s+v?(\d{1,4})/giu,

        /(?:^|[^\p{L}\p{N}_])[\p{L}][\p{L}\p{N}.+#_-]{2,}\s+v?(\d{1,4})(?=\s+(?:release|version|stable|lts|rc|beta|alpha|official|docs?|documentation|релиз\p{L}*|верси\p{L}*|стабильн\p{L}*|официальн\p{L}*|документац\p{L}*))/giu,

        /\/(?:release|releases|version|versions)\/v?(\d{1,4})(?=\/|$|[?#.-])/gi

    ];


    for(
        const pattern
        of patterns
    ){

        for(
            const match
            of source.matchAll(
                pattern
            )
        ){

            const value =
                safeString(
                    match?.[1]
                )
                    .toLowerCase();


            if(
                value
            ){

                values.add(
                    value
                );

            }

        }

    }


    return [
        ...values
    ];

}


function extractConcreteValues(
    text
) {

    const source =
        safeString(
            text
        );


    if(
        !source
    ){

        return [];

    }


    const values =
        new Set();


    const patterns = [

        /\bv?\d{1,3}(?:\.\d{1,4}){1,3}(?:[-+][0-9a-z.-]+)?\b/gi,

        /\b\d{4}-\d{2}-\d{2}\b/g,

        /\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b/g,

        /\b\d{5,}\b/g

    ];


    for(
        const pattern
        of patterns
    ){

        const matches =
            source.match(
                pattern
            );


        if(
            !matches
        ){

            continue;

        }


        for(
            const match
            of matches
        ){

            values.add(
                match.toLowerCase()
            );

        }

    }


    for(
        const value
        of extractContextualMajorVersions(
            source
        )
    ){

        values.add(
            value
        );

    }


    return [
        ...values
    ];

}


function escapeRegExp(
    value
) {

    return safeString(
        value
    )
        .replace(
            /[-/\\^$*+?.()|[\]{}]/g,
            "\\$&"
        );

}


function groundedTextContainsValue(
    groundedText,
    value
) {

    const cleanValue =
        safeString(
            value
        )
            .toLowerCase();


    if(
        !cleanValue
    ){

        return false;

    }


    const variants =
        new Set([
            cleanValue
        ]);


    if(
        /^v\d/i.test(
            cleanValue
        )
    ){

        variants.add(
            cleanValue.slice(1)
        );

    }
    else if(
        /^\d{1,4}(?:\.\d+)*$/i.test(
            cleanValue
        )
    ){

        variants.add(
            "v" + cleanValue
        );

    }


    for(
        const variant
        of variants
    ){

        const escaped =
            escapeRegExp(
                variant
            );


        const pattern =
            new RegExp(

                "(^|[^0-9a-zа-яё])"
                +
                escaped
                +
                "($|[^0-9a-zа-яё])",

                "iu"

            );


        if(
            pattern.test(
                groundedText
            )
        ){

            return true;

        }

    }


    return false;

}


/*
 * =========================================================
 * REPLAN CONSTRAINTS
 * =========================================================
 */


function getReplanConstraints(
    context
) {

    const constraints =
        context
            ?.metadata
            ?.replanConstraints;


    if(
        !constraints
        ||
        typeof constraints !== "object"
        ||
        Array.isArray(
            constraints
        )
    ){

        return {};
    }


    return constraints;

}


/*
 * =========================================================
 * FORBIDDEN QUERY
 * =========================================================
 */


function validateSearchQueryReuse(
    plan,
    context
) {

    const constraints =

        getReplanConstraints(
            context
        );


    const forbidden =

        new Set(

            uniqueStrings([

                ...safeArray(
                    context
                        ?.metadata
                        ?.triedSearchQueries
                ),

                ...safeArray(
                    constraints
                        ?.forbiddenSearchQueries
                )

            ])
                .map(
                    normalizeQuery
                )
                .filter(Boolean)

        );


    if(
        forbidden.size === 0
    ){

        return {
            success:
                true
        };

    }


    for(
        const step
        of safeArray(
            plan?.steps
        )
    ){

        if(
            safeString(
                step?.tool
            )
                .toLowerCase()
            !==
            "web_search"
        ){

            continue;

        }


        const query =

            normalizeQuery(
                step?.arguments?.query
            );


        if(
            query
            &&
            forbidden.has(
                query
            )
        ){

            return {

                success:
                    false,

                text:

                    (
                        "Planner повторил уже испробованный поисковый запрос: "
                        +
                        `"${safeString(step?.arguments?.query)}". `
                        +
                        "Сформируй содержательно другой поисковый запрос."
                    )

            };

        }

    }


    return {
        success:
            true
    };

}


/*
 * =========================================================
 * FORBIDDEN STATIC URL
 * =========================================================
 */


function validateStaticSourceReuse(
    plan,
    context
) {

    const constraints =

        getReplanConstraints(
            context
        );


    const forbidden =

        new Set(

            uniqueStrings([

                ...safeArray(
                    context
                        ?.metadata
                        ?.triedSourceUrls
                ),

                ...safeArray(
                    constraints
                        ?.forbiddenSourceUrls
                )

            ])
                .map(
                    normalizeUrl
                )
                .filter(Boolean)

        );


    if(
        forbidden.size === 0
    ){

        return {
            success:
                true
        };

    }


    for(
        const step
        of safeArray(
            plan?.steps
        )
    ){

        if(
            safeString(
                step?.tool
            )
                .toLowerCase()
            !==
            "web_fetch"
        ){

            continue;

        }


        /*
         * $from URL проверяется позже,
         * в Source Selector.
         */


        if(
            typeof step?.arguments?.url
            !==
            "string"
        ){

            continue;

        }


        const url =

            normalizeUrl(
                step.arguments.url
            );


        if(
            url
            &&
            forbidden.has(
                url
            )
        ){

            return {

                success:
                    false,

                text:

                    (
                        "Planner повторно использовал уже отклонённый источник: "
                        +
                        `"${step.arguments.url}". `
                        +
                        "Выбери другой источник."
                    )

            };

        }

    }


    return {
        success:
            true
    };

}


/*
 * =========================================================
 * UNKNOWN VALUES
 * =========================================================
 */


function validateUnknownConcreteValues(
    plan,
    context,
    task
) {

    const groundedText =

        buildGroundedText(
            task,
            context
        );


    /*
     * Проверяем только аргументы,
     * реально влияющие на внешний маршрут.
     */


    for(
        const step
        of safeArray(
            plan?.steps
        )
    ){

        const tool =

            safeString(
                step?.tool
            )
                .toLowerCase();


        let text =
            "";


        if(
            tool === "web_search"
        ){

            text =

                safeString(
                    step?.arguments?.query
                );

        }


        if(
            tool === "web_fetch"
            &&
            typeof step?.arguments?.url === "string"
        ){

            text =

                safeString(
                    step.arguments.url
                );

        }


        if(
            !text
        ){

            continue;

        }


        const concreteValues =

            extractConcreteValues(
                text
            );


        for(
            const value
            of concreteValues
        ){

            if(
                groundedTextContainsValue(
                    groundedText,
                    value
                )
            ){

                continue;

            }


            return {

                success:
                    false,

                text:

                    (
                        "Planner использовал неподтверждённое конкретное значение "
                        +
                        `"${value}" в аргументах ${tool}. `
                        +
                        "Это значение отсутствует в исходной задаче и подтверждённом контексте. "
                        +
                        "Сначала найди требуемый факт без предположения его значения."
                    )

            };

        }

    }


    return {
        success:
            true
    };

}


/*
 * =========================================================
 * PUBLIC POLICY
 * =========================================================
 */


export function validatePlannerGrounding({

    plan,

    context = {},

    task = ""

} = {}) {

    const queryReuse =

        validateSearchQueryReuse(
            plan,
            context
        );


    if(
        !queryReuse.success
    ){

        return queryReuse;

    }


    const sourceReuse =

        validateStaticSourceReuse(
            plan,
            context
        );


    if(
        !sourceReuse.success
    ){

        return sourceReuse;

    }


    const unknownValues =

        validateUnknownConcreteValues(

            plan,

            context,

            task

        );


    if(
        !unknownValues.success
    ){

        return unknownValues;

    }


    return {
        success:
            true
    };

          }
