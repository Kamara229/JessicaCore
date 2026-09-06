/*
 * =========================================================
 * JESSICA COMPOSER CONTEXT
 * =========================================================
 *
 * Подготавливает безопасный фактический контекст
 * для Answer Composer.
 *
 * Composer НЕ должен видеть:
 *
 * - названия внутренних tools;
 * - tool arguments;
 * - технические шаги Planner;
 * - $from;
 * - внутренний routing.
 *
 * Ему нужны только:
 *
 * - задача пользователя;
 * - требования к evidence;
 * - фактически полученные данные.
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_TOTAL_CONTEXT =
    20000;


const MAX_SOURCE_CONTENT =
    12000;


const MAX_GENERIC_DATA =
    6000;


/*
 * =========================================================
 * TEXT LIMIT
 * =========================================================
 */


function limitText(
    value,
    maxLength
) {

    const text =
        String(
            value || ""
        ).trim();


    if (
        text.length <=
        maxLength
    ) {

        return text;

    }


    return (
        text.slice(
            0,
            maxLength
        ) +
        "\n[Содержимое сокращено]"
    );

}


/*
 * =========================================================
 * WEB SOURCE
 * =========================================================
 */


function formatWebSource(
    result
) {

    const data =
        result?.data;


    if (
        !data ||
        typeof data !== "object"
    ) {

        return null;

    }


    const url =
        String(
            data.url || ""
        ).trim();


    const title =
        String(
            data.title || ""
        ).trim();


    const content =
        String(
            data.content || ""
        ).trim();


    if (!content) {

        return null;

    }


    return [
        "ИСТОЧНИК:",

        title
            ? `Название: ${title}`
            : null,

        url
            ? `URL: ${url}`
            : null,

        "",
        "Содержимое:",
        limitText(
            content,
            MAX_SOURCE_CONTENT
        )

    ]
        .filter(
            value =>
                value !== null
        )
        .join(
            "\n"
        );

}


/*
 * =========================================================
 * SEARCH RESULTS
 * =========================================================
 */


function formatSearchResults(
    result
) {

    const results =
        result?.data?.results;


    if (
        !Array.isArray(
            results
        ) ||
        results.length === 0
    ) {

        return null;

    }


    const cleanResults =
        results
            .slice(
                0,
                8
            )
            .map(
                item => ({
                    title:
                        item?.title || "",

                    url:
                        item?.url || "",

                    snippet:
                        item?.snippet || ""
                })
            );


    return [
        "РЕЗУЛЬТАТЫ ПОИСКА:",
        limitText(
            JSON.stringify(
                cleanResults,
                null,
                2
            ),
            MAX_GENERIC_DATA
        )
    ].join(
        "\n"
    );

}


/*
 * =========================================================
 * GENERIC RESULT
 * =========================================================
 */


function formatGenericResult(
    result
) {

    const text =
        String(
            result?.text || ""
        ).trim();


    const data =
        result?.data;


    if (
        !text &&
        (
            data === undefined ||
            data === null
        )
    ) {

        return null;

    }


    const parts =
        [
            "ПОЛУЧЕННЫЕ ДАННЫЕ:"
        ];


    if (text) {

        parts.push(
            limitText(
                text,
                MAX_GENERIC_DATA
            )
        );

    }


    if (
        data !== undefined &&
        data !== null
    ) {

        let serialized;


        try {

            serialized =
                JSON.stringify(
                    data,
                    null,
                    2
                );

        } catch {

            serialized =
                String(
                    data
                );

        }


        parts.push(
            limitText(
                serialized,
                MAX_GENERIC_DATA
            )
        );

    }


    return parts.join(
        "\n"
    );

}


/*
 * =========================================================
 * FORMAT ONE RESULT
 * =========================================================
 */


function formatResult(
    result
) {

    if (
        result?.success !== true
    ) {

        return null;

    }


    /*
     * Сначала пытаемся распознать
     * содержимое загруженной страницы.
     */


    const webSource =
        formatWebSource(
            result
        );


    if (webSource) {

        return webSource;

    }


    /*
     * Затем поисковую выдачу.
     */


    const searchResults =
        formatSearchResults(
            result
        );


    if (searchResults) {

        return searchResults;

    }


    /*
     * Остальные результаты.
     */


    return formatGenericResult(
        result
    );

}


/*
 * =========================================================
 * BUILD FACTUAL CONTEXT
 * =========================================================
 */


function buildFactualContext(
    taskRunResult
) {

    const results =
        Array.isArray(
            taskRunResult?.results
        )
            ? taskRunResult.results
            : [];


    const sections =
        results
            .map(
                formatResult
            )
            .filter(Boolean);


    if (
        sections.length === 0
    ) {

        return (
            "Дополнительные фактические данные " +
            "не были получены."
        );

    }


    return limitText(
        sections.join(
            "\n\n--------------------\n\n"
        ),
        MAX_TOTAL_CONTEXT
    );

}


/*
 * =========================================================
 * BUILD COMPOSER CONTEXT
 * =========================================================
 */


export function buildComposerContext(
    task,
    plan,
    taskRunResult
) {

    const evidenceMode =
        String(
            plan?.evidence?.mode ||
            "none"
        );


    const evidenceReason =
        String(
            plan?.evidence?.reason ||
            ""
        ).trim();


    const factualContext =
        buildFactualContext(
            taskRunResult
        );


    const sections =
        [
            "ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:",
            String(
                task || ""
            ).trim(),

            "",
            "ТРЕБОВАНИЯ К ДОКАЗАТЕЛЬСТВАМ:",
            `Режим: ${evidenceMode}`,

            evidenceReason
                ? `Цель: ${evidenceReason}`
                : null,

            "",
            "ФАКТИЧЕСКИЕ ДАННЫЕ:",
            factualContext
        ];


    return sections
        .filter(
            value =>
                value !== null
        )
        .join(
            "\n"
        );

      }
