/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY INSTRUCTIONS
 * =========================================================
 *
 * Формирует инструкции для Planner
 * после неудачной попытки выполнения.
 *
 *
 * Получает:
 *
 * - текущий PlanningContext;
 * - причину ошибки;
 * - компактную Retry History.
 *
 *
 * Возвращает:
 *
 * - обновлённый массив instructions.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает Planner;
 * - вызывает AI;
 * - изменяет metadata;
 * - выполняет инструменты;
 * - хранит историю самостоятельно.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_REASON_LENGTH =
    700;


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * UNIQUE STRINGS
 * =========================================================
 */


function uniqueStrings(
    values
) {

    if (
        !Array.isArray(
            values
        )
    ) {

        return [];

    }


    return [

        ...new Set(

            values

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
 * LIMIT TEXT
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


    if (
        text.length <= maxLength
    ) {

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
 * RETRY INSTRUCTION DETECTION
 * =========================================================
 *
 * PlanningContext возвращается между Replan.
 *
 * Поэтому старые retry-инструкции нужно удалять,
 * иначе после нескольких попыток получится:
 *
 * причина 1
 * причина 2
 * причина 3
 * ...
 *
 * Здесь удаляются только инструкции,
 * созданные этим модулем.
 *
 * Experience/User instructions сохраняются.
 *
 * =========================================================
 */


function isRetryInstruction(
    instruction
) {

    const text =
        safeString(
            instruction
        );


    if (!text) {

        return false;

    }


    const prefixes = [

        "Предыдущий маршрут выполнения",

        "Тип предыдущей ошибки:",

        "Причина предыдущей ошибки:",

        "Уже использованные поисковые запросы:",

        "В предыдущей попытке использовались инструменты:",

        "Не повторяй уже неудачный маршрут",

        "Сформируй другой поисковый запрос",

        "Измени стратегию выполнения",

        "Предыдущий поисковый маршрут"

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
 * CLEAN BASE INSTRUCTIONS
 * =========================================================
 */


function getBaseInstructions(
    planningContext
) {

    if (
        !Array.isArray(
            planningContext?.instructions
        )
    ) {

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
 * FORMAT QUERIES
 * =========================================================
 */


function formatSearchQueries(
    queries
) {

    if (
        !Array.isArray(
            queries
        ) ||
        queries.length === 0
    ) {

        return "";

    }


    const normalized =
        uniqueStrings(
            queries
        );


    if (
        normalized.length === 0
    ) {

        return "";

    }


    return normalized

        .map(
            query =>
                `"${query}"`
        )

        .join("; ");

}


/*
 * =========================================================
 * FORMAT TOOLS
 * =========================================================
 */


function formatTools(
    tools
) {

    if (
        !Array.isArray(
            tools
        ) ||
        tools.length === 0
    ) {

        return "";

    }


    return uniqueStrings(
        tools
    )
        .join(", ");

}


/*
 * =========================================================
 * SEARCH FAILURE INSTRUCTIONS
 * =========================================================
 */


function buildSearchFailureInstructions(
    failureType
) {

    if (
        failureType !== "no-search-results" &&
        failureType !== "no-suitable-source"
    ) {

        return [];

    }


    return [

        (
            "Предыдущий поисковый маршрут не позволил " +
            "найти подходящий источник."
        ),

        (
            "Сформируй другой поисковый запрос: " +
            "не повторяй уже использованную формулировку без изменений."
        ),

        (
            "Измени стратегию выполнения, если простой повтор поиска " +
            "не способен устранить причину ошибки."
        )

    ];

}


/*
 * =========================================================
 * BUILD RETRY INSTRUCTIONS
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


    /*
     * =====================================================
     * FAILURE
     * =====================================================
     */


    const failureType =
        safeString(
            failureResult?.failureType
        );


    const reason =
        limitText(
            failureResult?.reason ||
            failureResult?.text
        );


    instructions.push(
        "Предыдущий маршрут выполнения не дал корректный результат."
    );


    if (
        failureType
    ) {

        instructions.push(
            `Тип предыдущей ошибки: ${failureType}`
        );

    }


    if (
        reason
    ) {

        instructions.push(
            `Причина предыдущей ошибки: ${reason}`
        );

    }


    /*
     * =====================================================
     * SEARCH HISTORY
     * =====================================================
     */


    const triedSearchQueries =
        formatSearchQueries(
            retryHistory?.triedSearchQueries
        );


    if (
        triedSearchQueries
    ) {

        instructions.push(
            (
                "Уже использованные поисковые запросы: " +
                triedSearchQueries
            )
        );

    }


    /*
     * =====================================================
     * USED TOOLS
     * =====================================================
     */


    const usedTools =
        formatTools(
            retryHistory
                ?.previousAttempt
                ?.usedTools
        );


    if (
        usedTools
    ) {

        instructions.push(
            (
                "В предыдущей попытке использовались инструменты: " +
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
     * FAILURE-SPECIFIC RULES
     * =====================================================
     */


    instructions.push(

        ...buildSearchFailureInstructions(
            failureType
        )

    );


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return uniqueStrings(
        instructions
    );

}
