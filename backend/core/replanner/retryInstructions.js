/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY INSTRUCTIONS v3
 * =========================================================
 *
 * Формирует инструкции Planner
 * после неудачного Execution.
 *
 *
 * v3:
 *
 * - hard diversification constraints;
 * - forbidden search queries;
 * - forbidden source URLs;
 * - rejected route summaries;
 * - source validation handling;
 * - защита от повторения semantic route.
 *
 * =========================================================
 */


const MAX_REASON_LENGTH =
    900;


const MAX_ROUTE_SUMMARY_LENGTH =
    500;


/*
 * =========================================================
 * SAFE
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

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
                    item =>
                        safeString(
                            item
                        )
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * LIMIT
 * =========================================================
 */


function limitText(
    value,
    maxLength = MAX_REASON_LENGTH
) {

    const text =

        safeString(
            value
        );


    if(
        text.length <= maxLength
    ){

        return text;

    }


    return (

        text.slice(
            0,
            maxLength
        )

        +

        "…"

    );

}


/*
 * =========================================================
 * INTERNAL RETRY INSTRUCTION
 * =========================================================
 */


function isRetryInstruction(
    instruction
) {

    const text =

        safeString(
            instruction
        );


    if(
        !text
    ){

        return false;

    }


    const prefixes = [

        "Предыдущий маршрут выполнения",

        "Тип предыдущей ошибки:",

        "Причина предыдущей ошибки:",

        "Предыдущий evidence.mode:",

        "Уже использованные поисковые запросы:",

        "Уже проверенные источники:",

        "В предыдущей попытке использовались инструменты:",

        "Не повторяй уже неудачный маршрут",

        "Не используй повторно источник",

        "Сформируй другой поисковый запрос",

        "Измени стратегию выполнения",

        "Предыдущий поисковый маршрут",

        "Предыдущий источник не содержал",

        "Найди другой источник",

        "Для новой попытки сохрани",

        "Не считай официальный домен",

        "ОБЯЗАТЕЛЬНО сформируй новый Execution route.",

        "ЗАПРЕЩЕНО повторять поисковые запросы:",

        "ЗАПРЕЩЕНО повторно использовать источники:",

        "Отклонённые альтернативные маршруты:",

        "Новая стратегия должна отличаться",

        "Не считай изменением маршрута",

        "Не подставляй неподтверждённое"

    ];


    return prefixes.some(
        prefix =>
            text.startsWith(
                prefix
            )
    );

}


/*
 * =========================================================
 * BASE INSTRUCTIONS
 * =========================================================
 */


function getBaseInstructions(
    planningContext
) {

    if(
        !Array.isArray(
            planningContext?.instructions
        )
    ){

        return [];

    }


    return uniqueStrings(

        planningContext.instructions.filter(
            instruction =>
                !isRetryInstruction(
                    instruction
                )
        )

    );

}


/*
 * =========================================================
 * FORMAT
 * =========================================================
 */


function formatQuoted(
    values
) {

    const normalized =

        uniqueStrings(
            values
        );


    if(
        normalized.length === 0
    ){

        return "";

    }


    return normalized

        .map(
            value =>
                `"${value}"`
        )

        .join("; ");

}


function formatTools(
    tools
) {

    return uniqueStrings(
        tools
    )
        .join(", ");

}


/*
 * =========================================================
 * FAILURE DETAILS
 * =========================================================
 */


function getFailureDetails(
    failureResult
) {

    const details =
        failureResult?.details;


    if(
        !details
        ||
        typeof details !== "object"
        ||
        Array.isArray(
            details
        )
    ){

        return {};

    }


    return details;

}


/*
 * =========================================================
 * HARD DIVERSIFICATION
 * =========================================================
 */


function buildDiversificationInstructions({

    failureResult,

    retryHistory

}) {

    const details =

        getFailureDetails(
            failureResult
        );


    if(
        details.diversificationRequired
        !==
        true
    ){

        return [];

    }


    const forbiddenQueries =

        uniqueStrings([

            ...safeArray(
                retryHistory?.triedSearchQueries
            ),

            ...safeArray(
                details?.forbiddenSearchQueries
            )

        ]);


    const forbiddenUrls =

        uniqueStrings([

            ...safeArray(
                retryHistory?.triedSourceUrls
            ),

            ...safeArray(
                details?.forbiddenSourceUrls
            )

        ]);


    const rejectedRoutes =

        uniqueStrings(
            details?.rejectedRouteSummaries
        )
            .map(
                route =>
                    limitText(
                        route,
                        MAX_ROUTE_SUMMARY_LENGTH
                    )
            );


    const instructions = [

        (
            "ОБЯЗАТЕЛЬНО сформируй новый Execution route. " +
            "Предыдущая стратегия уже признана неуспешной."
        )

    ];


    /*
     * =====================================================
     * FORBIDDEN QUERIES
     * =====================================================
     */


    if(
        forbiddenQueries.length > 0
    ){

        instructions.push(

            (
                "ЗАПРЕЩЕНО повторять поисковые запросы: "
                +
                formatQuoted(
                    forbiddenQueries
                )
            )

        );

    }


    /*
     * =====================================================
     * FORBIDDEN URLS
     * =====================================================
     */


    if(
        forbiddenUrls.length > 0
    ){

        instructions.push(

            (
                "ЗАПРЕЩЕНО повторно использовать источники: "
                +
                formatQuoted(
                    forbiddenUrls
                )
            )

        );

    }


    /*
     * =====================================================
     * REJECTED ROUTES
     * =====================================================
     */


    if(
        rejectedRoutes.length > 0
    ){

        instructions.push(

            (
                "Отклонённые альтернативные маршруты: "
                +
                rejectedRoutes.join(
                    " || "
                )
            )

        );

    }


    /*
     * =====================================================
     * SEMANTIC DIVERSITY
     * =====================================================
     */


    instructions.push(

        (
            "Новая стратегия должна отличаться хотя бы по одному " +
            "содержательному параметру: поисковому запросу, целевому " +
            "источнику, типу страницы, последовательности действий " +
            "или способу получения требуемого факта."
        ),

        (
            "Не считай изменением маршрута только новый reasoningSummary, " +
            "другие имена step.id, перестановку слов без изменения смысла " +
            "или повтор тех же arguments."
        ),

        (
            "Не подставляй неподтверждённое конкретное значение " +
            "только для того, чтобы сформировать другой поисковый запрос."
        )

    );


    return instructions;

}


/*
 * =========================================================
 * SOURCE FAILURE DETECTION
 * =========================================================
 */


function isSourceValidationFailure({

    failureType,

    reason,

    retryHistory

}) {

    const normalizedType =

        safeString(
            failureType
        )
            .toLowerCase();


    const normalizedReason =

        safeString(
            reason
        )
            .toLowerCase();


    const previousEvidenceMode =

        safeString(

            retryHistory
                ?.previousAttempt
                ?.evidenceMode

        )
            .toLowerCase();


    if(
        normalizedType === "no-suitable-source"
        ||
        normalizedType === "source-validation-error"
        ||
        normalizedType === "source-content-error"
    ){

        return true;

    }


    if(
        previousEvidenceMode === "source_content"
        &&
        normalizedType === "validation-error"
    ){

        const patterns = [

            "не содержит",

            "не отображает",

            "недостаточ",

            "нет конкрет",

            "не подтверж",

            "требуется страница",

            "другой источник",

            "другая страница",

            "source",

            "page",

            "content"

        ];


        return patterns.some(
            pattern =>
                normalizedReason.includes(
                    pattern
                )
        );

    }


    return false;

}


/*
 * =========================================================
 * SEARCH FAILURE
 * =========================================================
 */


function buildSearchFailureInstructions(
    failureType
) {

    if(
        failureType !== "no-search-results"
        &&
        failureType !== "no-suitable-source"
    ){

        return [];

    }


    return [

        (
            "Предыдущий поисковый маршрут не позволил " +
            "найти подходящий источник."
        ),

        (
            "Сформируй другой поисковый запрос: " +
            "измени смысловую стратегию поиска, а не только порядок слов."
        ),

        (
            "Измени стратегию выполнения, если простой повтор поиска " +
            "не способен устранить причину ошибки."
        )

    ];

}


/*
 * =========================================================
 * SOURCE FAILURE
 * =========================================================
 */


function buildSourceFailureInstructions({

    failureType,

    reason,

    retryHistory

}) {

    if(
        !isSourceValidationFailure({

            failureType,

            reason,

            retryHistory

        })
    ){

        return [];

    }


    return [

        (
            "Предыдущий источник не содержал фактических данных, " +
            "необходимых для подтверждения ответа."
        ),

        (
            "Не используй повторно источник, который уже был проверен " +
            "и признан недостаточным."
        ),

        (
            "Найди другой источник или более конкретную страницу, " +
            "где требуемый факт непосредственно присутствует."
        ),

        (
            "Не считай официальный домен достаточным доказательством сам по себе: " +
            "страница должна подтверждать именно требуемое утверждение."
        ),

        (
            "Для новой попытки сохрани evidence.mode=source_content " +
            "и обязательно выполни web_fetch подходящего источника."
        )

    ];

}


/*
 * =========================================================
 * BUILD
 * =========================================================
 */


export function buildRetryInstructions({

    planningContext = {},

    failureResult = {},

    retryHistory = {}

} = {}) {

    /*
     * =====================================================
     * BASE
     * =====================================================
     */


    const instructions =

        getBaseInstructions(
            planningContext
        );


    const failureType =

        safeString(
            failureResult?.failureType
        );


    const reason =

        limitText(

            failureResult?.reason

            ||

            failureResult?.text

        );


    /*
     * =====================================================
     * CORE FAILURE
     * =====================================================
     */


    instructions.push(

        "Предыдущий маршрут выполнения не дал корректный результат."

    );


    if(
        failureType
    ){

        instructions.push(

            `Тип предыдущей ошибки: ${failureType}`

        );

    }


    if(
        reason
    ){

        instructions.push(

            `Причина предыдущей ошибки: ${reason}`

        );

    }


    /*
     * =====================================================
     * HARD DIVERSIFICATION
     * =====================================================
     *
     * Намеренно ставим ДО обычной истории,
     * чтобы эти правила первыми попали
     * в ограниченный Planning Context.
     *
     * =====================================================
     */


    instructions.push(

        ...buildDiversificationInstructions({

            failureResult,

            retryHistory

        })

    );


    /*
     * =====================================================
     * EVIDENCE MODE
     * =====================================================
     */


    const previousEvidenceMode =

        safeString(

            retryHistory
                ?.previousAttempt
                ?.evidenceMode

        );


    if(
        previousEvidenceMode
    ){

        instructions.push(

            `Предыдущий evidence.mode: ${previousEvidenceMode}`

        );

    }


    /*
     * =====================================================
     * HISTORY
     * =====================================================
     */


    const triedSearchQueries =

        formatQuoted(
            retryHistory?.triedSearchQueries
        );


    if(
        triedSearchQueries
    ){

        instructions.push(

            (
                "Уже использованные поисковые запросы: "
                +
                triedSearchQueries
            )

        );

    }


    const triedSourceUrls =

        formatQuoted(
            retryHistory?.triedSourceUrls
        );


    if(
        triedSourceUrls
    ){

        instructions.push(

            (
                "Уже проверенные источники: "
                +
                triedSourceUrls
            )

        );

    }


    const usedTools =

        formatTools(

            retryHistory
                ?.previousAttempt
                ?.usedTools

        );


    if(
        usedTools
    ){

        instructions.push(

            (
                "В предыдущей попытке использовались инструменты: "
                +
                usedTools
            )

        );

    }


    /*
     * =====================================================
     * CORE RETRY RULE
     * =====================================================
     */


    instructions.push(

        (
            "Не повторяй уже неудачный маршрут без существенных изменений."
        )

    );


    /*
     * =====================================================
     * FAILURE-SPECIFIC
     * =====================================================
     */


    instructions.push(

        ...buildSearchFailureInstructions(
            failureType
        )

    );


    instructions.push(

        ...buildSourceFailureInstructions({

            failureType,

            reason,

            retryHistory

        })

    );


    return uniqueStrings(
        instructions
    );

}
