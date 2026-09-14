/*
 * =========================================================
 * JESSICA LEARNING ANALYZER VALIDATOR
 * =========================================================
 *
 * Проверяет структуру результата
 * Learning Analyzer.
 *
 *
 * Проверяются:
 *
 * - analysis;
 * - understanding;
 * - reusable;
 * - clarificationQuestions;
 * - unsupportedSuggestions;
 * - proposedExperience;
 * - groundingEvidence.
 *
 *
 * ВАЖНО:
 *
 * Этот Validator проверяет только
 * структуру и типы данных.
 *
 * Реальное наличие evidence
 * в correction пользователя
 * проверяет отдельно:
 *
 * learningGroundingValidator.js
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - парсит JSON;
 * - проверяет цитаты по correction;
 * - создаёт Learning Proposal;
 * - сохраняет Experience;
 * - работает с Supabase.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONSTANTS
 * =========================================================
 */


const GROUNDED_FIELDS =
    [

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
     * GROUNDING EVIDENCE
     * =====================================================
     *
     * Каждая запись:
     *
     * {
     *   field,
     *   value,
     *   evidence
     * }
     *
     * Здесь проверяем только структуру.
     *
     * =====================================================
     */


    if (
        !Array.isArray(
            analysis.groundingEvidence
        )
    ) {

        errors.push(
            "groundingEvidence должен быть массивом"
        );

    } else {


        for (
            let index = 0;
            index < analysis.groundingEvidence.length;
            index += 1
        ) {


            const item =
                analysis.groundingEvidence[
                    index
                ];


            /*
             * =============================================
             * ITEM OBJECT
             * =============================================
             */


            if (
                !isObject(
                    item
                )
            ) {

                errors.push(
                    `groundingEvidence[${index}] должен быть объектом`
                );


                continue;

            }


            /*
             * =============================================
             * FIELD
             * =============================================
             */


            if (
                !isNonEmptyString(
                    item.field
                )
            ) {

                errors.push(
                    `groundingEvidence[${index}].field не указан`
                );

            } else if (
                !GROUNDED_FIELDS.includes(
                    item.field.trim()
                )
            ) {

                errors.push(
                    `groundingEvidence[${index}].field содержит недопустимое значение`
                );

            }


            /*
             * =============================================
             * VALUE
             * =============================================
             */


            if (
                !isNonEmptyString(
                    item.value
                )
            ) {

                errors.push(
                    `groundingEvidence[${index}].value не указан`
                );

            }


            /*
             * =============================================
             * EVIDENCE
             * =============================================
             */


            if (
                !isNonEmptyString(
                    item.evidence
                )
            ) {

                errors.push(
                    `groundingEvidence[${index}].evidence не указан`
                );

            }

        }

    }


    /*
     * =====================================================
     * GROUNDING REQUIRED FOR RULES
     * =====================================================
     *
     * На структурном уровне проверяем,
     * что количество evidence хотя бы
     * потенциально может покрыть
     * все операционные правила.
     *
     *
     * Точное соответствие:
     *
     * field + value + evidence
     *
     * проверит Grounding Validator.
     *
     * =====================================================
     */


    if (
        isObject(
            analysis.proposedExperience
        ) &&
        Array.isArray(
            analysis.groundingEvidence
        )
    ) {


        const proposed =
            analysis.proposedExperience;


        const proposedRulesCount =
            GROUNDED_FIELDS.reduce(
                (
                    total,
                    field
                ) => {


                    return (
                        total +
                        (
                            Array.isArray(
                                proposed[field]
                            )
                                ? proposed[field].length
                                : 0
                        )
                    );


                },
                0
            );


        if (
            analysis.groundingEvidence.length <
            proposedRulesCount
        ) {

            errors.push(
                "groundingEvidence не покрывает все операционные правила proposedExperience"
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
