/*
 * =========================================================
 * JESSICA LEARNING ANALYZER VALIDATOR
 * =========================================================
 *
 * Проверяет структуру результата
 * Learning Analyzer.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - парсит JSON;
 * - создаёт Learning Proposal;
 * - сохраняет Experience;
 * - подтверждает обучение;
 * - работает с Supabase.
 *
 * =========================================================
 */


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


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


function isNonEmptyString(
    value
) {

    return (
        typeof value === "string" &&
        value.trim().length > 0
    );

}


function isStringArray(
    value
) {

    return (
        Array.isArray(
            value
        ) &&
        value.every(
            item =>
                typeof item === "string"
        )
    );

}


/*
 * =========================================================
 * VALIDATE LEARNING ANALYSIS
 * =========================================================
 */


export function validateLearningAnalysis(
    analysis
) {


    const errors =
        [];


    /*
     * =====================================================
     * ROOT
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

            errors: [
                "Learning Analysis должен быть объектом"
            ]

        };

    }


    /*
     * =====================================================
     * ANALYSIS BLOCK
     * =====================================================
     */


    if (
        !isObject(
            analysis.analysis
        )
    ) {

        errors.push(
            "Отсутствует блок analysis"
        );

    } else {


        if (
            !isNonEmptyString(
                analysis.analysis.errorSummary
            )
        ) {

            errors.push(
                "analysis.errorSummary не указан"
            );

        }


        if (
            !isNonEmptyString(
                analysis.analysis.correctionSummary
            )
        ) {

            errors.push(
                "analysis.correctionSummary не указан"
            );

        }


        if (
            !isNonEmptyString(
                analysis.analysis.differenceSummary
            )
        ) {

            errors.push(
                "analysis.differenceSummary не указан"
            );

        }

    }


    /*
     * =====================================================
     * UNDERSTANDING
     * =====================================================
     */


    if (
        !isNonEmptyString(
            analysis.understanding
        )
    ) {

        errors.push(
            "understanding не указан"
        );

    }


    /*
     * =====================================================
     * REUSABLE
     * =====================================================
     */


    if (
        typeof analysis.reusable !==
        "boolean"
    ) {

        errors.push(
            "reusable должен быть boolean"
        );

    }


    /*
     * =====================================================
     * CLARIFICATION QUESTIONS
     * =====================================================
     */


    if (
        !isStringArray(
            analysis.clarificationQuestions
        )
    ) {

        errors.push(
            "clarificationQuestions должен быть массивом строк"
        );

    }


    /*
     * =====================================================
     * UNSUPPORTED SUGGESTIONS
     * =====================================================
     *
     * Здесь Analyzer может указать идеи,
     * которые считает полезными,
     * но которые НЕ были подтверждены
     * исправлением пользователя.
     *
     * Они никогда не должны автоматически
     * становиться частью Skill.
     *
     * =====================================================
     */


    if (
        !isStringArray(
            analysis.unsupportedSuggestions
        )
    ) {

        errors.push(
            "unsupportedSuggestions должен быть массивом строк"
        );

    }


    /*
     * =====================================================
     * PROPOSED EXPERIENCE
     * =====================================================
     */


    if (
        !isObject(
            analysis.proposedExperience
        )
    ) {

        errors.push(
            "proposedExperience отсутствует"
        );

    } else {


        const proposed =
            analysis.proposedExperience;


        /*
         * =================================================
         * NAME
         * =================================================
         */


        if (
            !isNonEmptyString(
                proposed.name
            )
        ) {

            errors.push(
                "proposedExperience.name не указан"
            );

        }


        /*
         * =================================================
         * DESCRIPTION
         * =================================================
         */


        if (
            !isNonEmptyString(
                proposed.description
            )
        ) {

            errors.push(
                "proposedExperience.description не указан"
            );

        }


        /*
         * =================================================
         * ARRAY FIELDS
         * =================================================
         */


        const arrayFields =
            [

                "taskTypes",

                "keywords",

                "tags",

                "strategy",

                "sourcePriority",

                "validationRules",

                "failurePatterns"

            ];


        for (
            const field
            of arrayFields
        ) {


            if (
                !isStringArray(
                    proposed[field]
                )
            ) {

                errors.push(
                    `proposedExperience.${field} должен быть массивом строк`
                );

            }

        }


        /*
         * =================================================
         * REUSABLE STRATEGY
         * =================================================
         *
         * Если исправление признано
         * переиспользуемым,
         * Skill обязан содержать
         * хотя бы один шаг стратегии.
         *
         * =================================================
         */


        if (
            analysis.reusable === true &&
            (
                !Array.isArray(
                    proposed.strategy
                ) ||
                proposed.strategy.length === 0
            )
        ) {

            errors.push(
                "Для reusable Skill strategy не должна быть пустой"
            );

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

        errors

    };

}
