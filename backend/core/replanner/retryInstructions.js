/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY INSTRUCTIONS v2
 * =========================================================
 *
 * Формирует инструкции Planner
 * после неудачного Execution.
 *
 *
 * v2:
 *
 * - учитывает rejected source URLs;
 * - распознаёт source-validation failure;
 * - запрещает повторно использовать
 *   недостаточный источник;
 * - заставляет искать источник,
 *   содержащий именно нужный факт;
 * - сохраняет требуемый evidence strength.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_REASON_LENGTH =
    900;


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

    if(
        !Array.isArray(
            values
        )
    ){

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
 * RETRY INSTRUCTION DETECTION
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

        "Уже использованные поисковые запросы:",

        "Уже проверенные источники:",

        "В предыдущей попытке использовались инструменты:",

        "Предыдущий evidence.mode:",

        "Не повторяй уже неудачный маршрут",

        "Не используй повторно источник",

        "Сформируй другой поисковый запрос",

        "Измени стратегию выполнения",

        "Предыдущий поисковый маршрут",

        "Предыдущий источник не содержал",

        "Найди другой источник",

        "Для новой попытки сохрани",

        "Не считай официальный домен"

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
 * FORMAT VALUES
 * =========================================================
 */


function formatQuoted(
    values
) {

    if(
        !Array.isArray(values)
        ||
        values.length === 0
    ){

        return "";

    }


    return uniqueStrings(
        values
    )
        .map(
            value =>
                `"${value}"`
        )
        .join("; ");

}


function formatTools(
    tools
) {

    if(
        !Array.isArray(tools)
        ||
        tools.length === 0
    ){

        return "";

    }


    return uniqueStrings(
        tools
    )
        .join(", ");

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


    /*
     * Явные типы.
     */


    if(
        normalizedType === "no-suitable-source"
        ||
        normalizedType === "source-validation-error"
        ||
        normalizedType === "source-content-error"
    ){

        return true;

    }


    /*
     * Если предыдущий Plan требовал source_content,
     * а Validator сообщил, что содержимого
     * недостаточно — это source failure,
     * даже если общий failureType = validation-error.
     */


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


        if(
            patterns.some(
                pattern =>
                    normalizedReason.includes(
                        pattern
                    )
            )
        ){

            return true;

        }

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
 * SOURCE FAILURE INSTRUCTIONS
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


    const instructions = [

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
            "где требуемый факт явно присутствует в содержимом."
        ),

        (
            "Не считай официальный домен достаточным доказательством сам по себе: " +
            "страница должна непосредственно подтверждать требуемое утверждение."
        ),

        (
            "Для новой попытки сохрани evidence.mode=source_content " +
            "и обязательно выполни web_fetch подходящего источника."
        )

    ];


    const previousUrls =

        formatQuoted(
            retryHistory?.triedSourceUrls
        );


    if(
        previousUrls
    ){

        instructions.push(

            (
                "Уже проверенные источники: "
                +
                previousUrls
            )

        );

    }


    return instructions;

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

            failureResult?.reason

            ||

            failureResult?.text

        );


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
     * PREVIOUS EVIDENCE MODE
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
     * SEARCH HISTORY
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


    /*
     * =====================================================
     * SOURCE HISTORY
     * =====================================================
     */


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
     * FAILURE-SPECIFIC RULES
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


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return uniqueStrings(
        instructions
    );

}
