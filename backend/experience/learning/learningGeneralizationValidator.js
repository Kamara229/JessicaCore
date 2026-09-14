/*
 * =========================================================
 * JESSICA LEARNING GENERALIZATION VALIDATOR
 * =========================================================
 *
 * Проверяет, что reusable Skill
 * не содержит конкретные данные
 * из единичного примера.
 *
 *
 * Например:
 *
 * задача:
 * "Найди официальный сайт Blender"
 *
 * correction:
 * "Для Blender правильный домен — blender.org"
 *
 *
 * blender.org полезен для текущей задачи,
 * но не должен становиться:
 *
 * - keyword общего Skill;
 * - tag;
 * - taskType;
 * - частью общей strategy;
 * - общим validationRule.
 *
 *
 * Этот Validator НЕ:
 *
 * - вызывает AI;
 * - определяет grounding;
 * - сохраняет Skill;
 * - изменяет Proposal;
 * - работает с Supabase.
 *
 * =========================================================
 */


/*
 * =========================================================
 * FIELDS
 * =========================================================
 */


const GENERALIZED_FIELDS =
    [

        "name",

        "description",

        "taskTypes",

        "keywords",

        "tags",

        "strategy",

        "sourcePriority",

        "validationRules",

        "failurePatterns"

    ];


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .replace(
            /\s+/g,
            " "
        )
        .trim()
        .toLowerCase();

}


function isObject(
    value
) {

    return (
        value &&
        typeof value === "object" &&
        !Array.isArray(
            value
        )
    );

}


/*
 * =========================================================
 * EXTRACT URLs
 * =========================================================
 */


function extractUrls(
    text
) {


    const value =
        String(
            text || ""
        );


    const matches =
        value.match(
            /https?:\/\/[^\s"'<>]+/gi
        );


    return Array.isArray(
        matches
    )
        ? matches
        : [];

}


/*
 * =========================================================
 * EXTRACT EMAILS
 * =========================================================
 */


function extractEmails(
    text
) {


    const value =
        String(
            text || ""
        );


    const matches =
        value.match(
            /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi
        );


    return Array.isArray(
        matches
    )
        ? matches
        : [];

}


/*
 * =========================================================
 * EXTRACT DOMAINS
 * =========================================================
 */


function extractDomains(
    text
) {


    const value =
        String(
            text || ""
        );


    const matches =
        value.match(
            /\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b/gi
        );


    return Array.isArray(
        matches
    )
        ? matches
        : [];

}


/*
 * =========================================================
 * EXTRACT LONG IDENTIFIERS
 * =========================================================
 *
 * Ловим очевидные конкретные ID:
 *
 * - длинные числовые значения;
 * - UUID.
 *
 * Короткие числа специально
 * не считаем специфичным примером.
 *
 * =========================================================
 */


function extractIdentifiers(
    text
) {


    const value =
        String(
            text || ""
        );


    const identifiers =
        [];


    const uuidMatches =
        value.match(
            /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi
        );


    if (
        Array.isArray(
            uuidMatches
        )
    ) {

        identifiers.push(
            ...uuidMatches
        );

    }


    const numericMatches =
        value.match(
            /\b\d{7,}\b/g
        );


    if (
        Array.isArray(
            numericMatches
        )
    ) {

        identifiers.push(
            ...numericMatches
        );

    }


    return identifiers;

}


/*
 * =========================================================
 * UNIQUE MARKERS
 * =========================================================
 */


function uniqueMarkers(
    values
) {


    const normalized =
        values
            .map(
                value =>
                    normalizeText(
                        value
                    )
            )
            .filter(
                Boolean
            );


    return [
        ...new Set(
            normalized
        )
    ];

}


/*
 * =========================================================
 * EXTRACT EXAMPLE MARKERS
 * =========================================================
 *
 * Источники:
 *
 * - исходная задача;
 * - correction;
 * - correctedAnswer.
 *
 *
 * Мы извлекаем только очевидные
 * конкретные технические значения.
 *
 * =========================================================
 */


function extractExampleMarkers({

    task,

    correction,

    correctedAnswer

} = {}) {


    const sourceText =
        [

            task,

            correction,

            correctedAnswer

        ]
            .map(
                value =>
                    String(
                        value || ""
                    )
            )
            .join(
                "\n"
            );


    return uniqueMarkers([

        ...extractUrls(
            sourceText
        ),

        ...extractEmails(
            sourceText
        ),

        ...extractDomains(
            sourceText
        ),

        ...extractIdentifiers(
            sourceText
        )

    ]);

}


/*
 * =========================================================
 * FIELD VALUES
 * =========================================================
 */


function getFieldValues(
    proposedExperience,
    field
) {


    const value =
        proposedExperience?.[
            field
        ];


    if (
        Array.isArray(
            value
        )
    ) {

        return value
            .map(
                item =>
                    String(
                        item || ""
                    ).trim()
            )
            .filter(
                Boolean
            );

    }


    if (
        typeof value === "string" &&
        value.trim()
    ) {

        return [
            value.trim()
        ];

    }


    return [];

}


/*
 * =========================================================
 * CREATE ERROR
 * =========================================================
 */


function createError({

    field,

    value,

    marker,

    message

}) {


    return {

        type:
            "example_specific_value",

        field:
            String(
                field || ""
            ),

        value:
            String(
                value || ""
            ),

        marker:
            String(
                marker || ""
            ),

        message:
            String(
                message ||
                "Обнаружено конкретное значение из единичного примера"
            )

    };

}


/*
 * =========================================================
 * VALIDATE GENERALIZATION
 * =========================================================
 */


export function validateLearningGeneralization({

    task,

    correction,

    correctedAnswer = "",

    analysis

} = {}) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if (
        !isObject(
            analysis
        )
    ) {

        return {

            valid:
                false,

            checked:
                false,

            exampleMarkers:
                [],

            errors: [

                {

                    type:
                        "input",

                    field:
                        "",

                    value:
                        "",

                    marker:
                        "",

                    message:
                        "Generalization Validator: Learning Analysis отсутствует"

                }

            ]

        };

    }


    /*
     * =====================================================
     * NON-REUSABLE
     * =====================================================
     *
     * Если Analyzer уже признал,
     * что правило не переиспользуемое,
     * общий Skill всё равно
     * не будет сохраняться.
     *
     * Поэтому дополнительная проверка
     * здесь не требуется.
     *
     * =====================================================
     */


    if (
        analysis.reusable !== true
    ) {

        return {

            valid:
                true,

            checked:
                false,

            reason:
                "Learning Analysis не является reusable",

            exampleMarkers:
                [],

            errors:
                []

        };

    }


    /*
     * =====================================================
     * PROPOSED EXPERIENCE
     * =====================================================
     */


    const proposedExperience =
        analysis.proposedExperience;


    if (
        !isObject(
            proposedExperience
        )
    ) {

        return {

            valid:
                false,

            checked:
                false,

            exampleMarkers:
                [],

            errors: [

                {

                    type:
                        "structure",

                    field:
                        "",

                    value:
                        "",

                    marker:
                        "",

                    message:
                        "Generalization Validator: proposedExperience отсутствует"

                }

            ]

        };

    }


    /*
     * =====================================================
     * EXAMPLE MARKERS
     * =====================================================
     */


    const exampleMarkers =
        extractExampleMarkers({

            task,

            correction,

            correctedAnswer

        });


    /*
     * Если конкретных технических
     * значений в примере нет,
     * блокировать нечего.
     */


    if (
        exampleMarkers.length === 0
    ) {

        return {

            valid:
                true,

            checked:
                true,

            exampleMarkers:
                [],

            errors:
                []

        };

    }


    /*
     * =====================================================
     * CHECK SKILL FIELDS
     * =====================================================
     */


    const errors =
        [];


    for (
        const field
        of GENERALIZED_FIELDS
    ) {


        const values =
            getFieldValues(
                proposedExperience,
                field
            );


        for (
            const value
            of values
        ) {


            const normalizedValue =
                normalizeText(
                    value
                );


            for (
                const marker
                of exampleMarkers
            ) {


                /*
                 * =================================================
                 * MARKER FOUND
                 * =================================================
                 *
                 * Например:
                 *
                 * keyword = "blender.org"
                 *
                 * или:
                 *
                 * strategy =
                 * "Использовать blender.org"
                 *
                 * =================================================
                 */


                if (
                    normalizedValue.includes(
                        marker
                    )
                ) {

                    errors.push(

                        createError({

                            field,

                            value,

                            marker,

                            message:
                                `Поле proposedExperience.${field} содержит конкретное значение "${marker}" из единичного примера`

                        })

                    );

                }

            }

        }

    }


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        valid:
            errors.length === 0,

        checked:
            true,

        exampleMarkers,

        checkedFields:
            [
                ...GENERALIZED_FIELDS
            ],

        errors

    };

}
