/*
 * =========================================================
 * JESSICA PLANNER
 * ATTEMPT FEEDBACK v1
 * =========================================================
 *
 * Semantic validation failure
 *        ↓
 * Rejected Plan
 *        ↓
 * Structured Repair Feedback
 *        ↓
 * Next Planner Attempt
 *
 *
 * Задача:
 *
 * превратить сухую ошибку Validator
 * в конкретную инструкцию ремонта.
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - валидирует Plan;
 * - изменяет PlanningContext;
 * - выполняет Retry;
 * - выполняет Tools.
 *
 * =========================================================
 */


/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


const MAX_REASON_LENGTH =
    1400;


const MAX_FEEDBACK_LENGTH =
    2800;


const MAX_REJECTED_QUERIES =
    5;


const MAX_REJECTED_URLS =
    5;


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
 * LIMIT TEXT
 * =========================================================
 */


function limitText(
    value,
    maxLength
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
 * REJECTED SEARCH QUERIES
 * =========================================================
 */


function extractSearchQueries(
    plan
) {

    return uniqueStrings(

        safeArray(
            plan?.steps
        )

            .filter(
                step =>
                    safeString(
                        step?.tool
                    )
                        .toLowerCase()
                    ===
                    "web_search"
            )

            .map(
                step =>
                    step?.arguments?.query
            )

    )
        .slice(
            0,
            MAX_REJECTED_QUERIES
        );

}


/*
 * =========================================================
 * REJECTED STATIC URLS
 * =========================================================
 */


function extractStaticUrls(
    plan
) {

    return uniqueStrings(

        safeArray(
            plan?.steps
        )

            .filter(
                step =>
                    safeString(
                        step?.tool
                    )
                        .toLowerCase()
                    ===
                    "web_fetch"
            )

            .map(
                step => {

                    const url =
                        step?.arguments?.url;


                    return typeof url === "string"
                        ? url
                        : "";

                }
            )

    )
        .slice(
            0,
            MAX_REJECTED_URLS
        );

}


/*
 * =========================================================
 * QUOTED VALUE
 * =========================================================
 */


function extractQuotedValue(
    reason
) {

    const text =

        safeString(
            reason
        );


    if(
        !text
    ){

        return "";
    }


    /*
     * Основной формат Grounding Policy:
     *
     * значение "16.3"
     */


    const quotedMatch =

        text.match(
            /(?:значение|источник|запрос)[^"]*"([^"]+)"/i
        );


    if(
        quotedMatch?.[1]
    ){

        return safeString(
            quotedMatch[1]
        );

    }


    return "";

}


/*
 * =========================================================
 * FAILURE TYPES
 * =========================================================
 */


function isUnknownConcreteValueFailure(
    reason
) {

    const text =

        safeString(
            reason
        )
            .toLowerCase();


    return (

        text.includes(
            "неподтверждённое конкретное значение"
        )

        ||

        text.includes(
            "неподтвержденное конкретное значение"
        )

        ||

        (
            text.includes(
                "отсутствует в исходной задаче"
            )

            &&

            text.includes(
                "подтверждённом контексте"
            )
        )

    );

}


function isRepeatedSearchFailure(
    reason
) {

    const text =

        safeString(
            reason
        )
            .toLowerCase();


    return (

        text.includes(
            "повторил уже испробованный поисковый запрос"
        )

        ||

        text.includes(
            "повторять поисковые запросы"
        )

    );

}


function isRepeatedSourceFailure(
    reason
) {

    const text =

        safeString(
            reason
        )
            .toLowerCase();


    return (

        text.includes(
            "повторно использовал уже отклонённый источник"
        )

        ||

        text.includes(
            "повторно использовать источники"
        )

    );

}


function isMissingFetchFailure(
    reason
) {

    const text =

        safeString(
            reason
        )
            .toLowerCase();


    return (

        text.includes(
            "source_content"
        )

        &&

        text.includes(
            "web_fetch"
        )

    );

}


/*
 * =========================================================
 * FORMAT FORBIDDEN
 * =========================================================
 */


function formatQuoted(
    values
) {

    return uniqueStrings(
        values
    )

        .map(
            value =>
                `"${value}"`
        )

        .join("; ");

}


/*
 * =========================================================
 * UNKNOWN VALUE REPAIR
 * =========================================================
 */


function buildUnknownValueRepair({

    reason,

    rejectedPlan

}) {

    const value =

        extractQuotedValue(
            reason
        );


    const queries =

        extractSearchQueries(
            rejectedPlan
        );


    const lines = [

        "ТИП ОШИБКИ: Planner заранее предположил значение факта, который задача требует сначала определить.",

        "",

        "ЖЁСТКОЕ ПРАВИЛО РЕМОНТА:",

        (
            "Не используй предполагаемое конкретное значение искомого факта "
            +
            "в новом search query, URL или arguments."
        ),

        (
            "Не заменяй отклонённое значение другим придуманным значением "
            +
            "того же неизвестного факта."
        ),

        (
            "Например, если неизвестен номер актуальной версии, "
            +
            "нельзя заменить один предполагаемый номер версии другим."
        ),

        (
            "Сформулируй поиск через сам искомый факт: "
            +
            "latest/current/stable release, официальный release page, "
            +
            "release history или эквивалентную нейтральную формулировку."
        ),

        (
            "Конкретное значение можно использовать только после того, "
            +
            "как оно пришло из разрешённого подтверждённого контекста."
        )

    ];


    if(
        value
    ){

        lines.push(

            "",

            (
                `ЗАПРЕЩЁННОЕ НЕПОДТВЕРЖДЁННОЕ ЗНАЧЕНИЕ: "${value}".`
            )

        );

    }


    if(
        queries.length > 0
    ){

        lines.push(

            "",

            (
                "НЕ КОПИРУЙ ОТКЛОНЁННЫЕ SEARCH QUERIES: "
                +
                formatQuoted(
                    queries
                )
            )

        );

    }


    return lines;

}


/*
 * =========================================================
 * REPEATED SEARCH REPAIR
 * =========================================================
 */


function buildRepeatedSearchRepair(
    rejectedPlan
) {

    const queries =

        extractSearchQueries(
            rejectedPlan
        );


    const lines = [

        "ТИП ОШИБКИ: повтор уже испробованного search route.",

        "",

        (
            "Следующий Plan обязан использовать содержательно другой "
            +
            "поисковый запрос."
        ),

        (
            "Не считай новым запросом изменение регистра, порядка слов "
            +
            "или косметическую переформулировку без изменения стратегии."
        )

    ];


    if(
        queries.length > 0
    ){

        lines.push(

            "",

            (
                "ОТКЛОНЁННЫЕ SEARCH QUERIES: "
                +
                formatQuoted(
                    queries
                )
            )

        );

    }


    return lines;

}


/*
 * =========================================================
 * REPEATED SOURCE REPAIR
 * =========================================================
 */


function buildRepeatedSourceRepair(
    rejectedPlan
) {

    const urls =

        extractStaticUrls(
            rejectedPlan
        );


    const lines = [

        "ТИП ОШИБКИ: повтор уже отклонённого источника.",

        "",

        "Следующий Plan должен использовать другой источник.",

        (
            "Если URL заранее неизвестен, используй новый web_search "
            +
            "и последующий web_fetch результата."
        )

    ];


    if(
        urls.length > 0
    ){

        lines.push(

            "",

            (
                "ОТКЛОНЁННЫЕ URL: "
                +
                formatQuoted(
                    urls
                )
            )

        );

    }


    return lines;

}


/*
 * =========================================================
 * SOURCE CONTENT REPAIR
 * =========================================================
 */


function buildMissingFetchRepair()
{

    return [

        "ТИП ОШИБКИ: нарушение Source Content Contract.",

        "",

        (
            "Если evidence.mode=source_content, Plan обязательно должен "
            +
            "содержать web_fetch."
        ),

        (
            "Если URL заранее неизвестен, используй web_search → web_fetch "
            +
            "с передачей URL через $from."
        )

    ];

}


/*
 * =========================================================
 * GENERIC REPAIR
 * =========================================================
 */


function buildGenericRepair()
{

    return [

        "Исправь именно причину Validation failure.",

        (
            "Не возвращай тот же Plan с изменённым reasoningSummary, "
            +
            "step.id или косметически изменёнными arguments."
        ),

        (
            "Следующий Plan должен пройти указанное машинное ограничение."
        )

    ];

}


/*
 * =========================================================
 * PUBLIC BUILDER
 * =========================================================
 */


export function buildPlannerAttemptFeedback({

    reason,

    rejectedPlan = null,

    attempt = 1

} = {}) {

    const cleanReason =

        limitText(

            reason,

            MAX_REASON_LENGTH

        )

        ||

        "Planner validation failed";


    const lines = [

        "ПРЕДЫДУЩИЙ EXECUTION PLAN ОТКЛОНЁН ВАЛИДАТОРОМ.",

        `Отклонённая попытка: ${attempt}.`,

        "",

        "ПРИЧИНА:",

        cleanReason,

        "",

        "Нужно создать НОВЫЙ корректный Execution Plan."

    ];


    /*
     * =====================================================
     * SPECIFIC REPAIR
     * =====================================================
     */


    if(
        isUnknownConcreteValueFailure(
            cleanReason
        )
    ){

        lines.push(

            "",

            ...buildUnknownValueRepair({

                reason:
                    cleanReason,

                rejectedPlan

            })

        );

    }
    else if(
        isRepeatedSearchFailure(
            cleanReason
        )
    ){

        lines.push(

            "",

            ...buildRepeatedSearchRepair(
                rejectedPlan
            )

        );

    }
    else if(
        isRepeatedSourceFailure(
            cleanReason
        )
    ){

        lines.push(

            "",

            ...buildRepeatedSourceRepair(
                rejectedPlan
            )

        );

    }
    else if(
        isMissingFetchFailure(
            cleanReason
        )
    ){

        lines.push(

            "",

            ...buildMissingFetchRepair()

        );

    }
    else{

        lines.push(

            "",

            ...buildGenericRepair()

        );

    }


    /*
     * =====================================================
     * FINAL CONTRACT
     * =====================================================
     */


    lines.push(

        "",

        "КРИТИЧЕСКО:",

        (
            "Не объясняй ошибку и не спорь с Validator. "
            +
            "Исправь Plan."
        ),

        (
            "Верни только новый JSON Execution Plan."
        )

    );


    return limitText(

        lines
            .filter(
                value =>
                    value !== null
                    &&
                    value !== undefined
            )
            .join("\n"),

        MAX_FEEDBACK_LENGTH

    );

}
