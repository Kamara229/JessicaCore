/*
 * =========================================================
 * JESSICA LEARNING GROUNDING VALIDATOR
 * =========================================================
 *
 * Проверяет, что операционные правила,
 * предложенные Learning Analyzer,
 * действительно имеют основание
 * в исправлении пользователя.
 *
 *
 * Проверяются поля:
 *
 * - strategy;
 * - sourcePriority;
 * - validationRules;
 * - failurePatterns.
 *
 *
 * Для каждого элемента должно существовать:
 *
 * {
 *   field,
 *   value,
 *   evidence
 * }
 *
 *
 * И evidence должен реально содержаться
 * в исходном correction пользователя.
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - сохраняет Experience;
 * - подтверждает Proposal;
 * - изменяет Skill;
 * - работает с Supabase.
 *
 * =========================================================
 */


/*
 * =========================================================
 * GROUNDED FIELDS
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
 * NORMALIZE TEXT
 * =========================================================
 *
 * Нормализация нужна только для сравнения:
 *
 * - регистр не имеет значения;
 * - лишние пробелы не имеют значения;
 * - переносы строк превращаются в пробелы.
 *
 * Смысл текста не изменяется.
 *
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


/*
 * =========================================================
 * IS OBJECT
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


/*
 * =========================================================
 * CREATE ERROR
 * =========================================================
 */


function createError({

    type,

    field = "",

    value = "",

    evidence = "",

    message

}) {

    return {

        type:
            String(
                type || "grounding"
            ),

        field:
            String(
                field || ""
            ),

        value:
            String(
                value || ""
            ),

        evidence:
            String(
                evidence || ""
            ),

        message:
            String(
                message ||
                "Ошибка Grounding Validation"
            )

    };

}


/*
 * =========================================================
 * VALIDATE LEARNING GROUNDING
 * =========================================================
 */


export function validateLearningGrounding({

    correction,

    analysis

} = {}) {


    const errors =
        [];


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    const normalizedCorrection =
        normalizeText(
            correction
        );


    if (!normalizedCorrection) {

        return {

            valid:
                false,

            errors: [

                createError({

                    type:
                        "input",

                    message:
                        "Grounding Validator: correction пользователя отсутствует"

                })

            ]

        };

    }


    if (
        !isObject(
            analysis
        )
    ) {

        return {

            valid:
                false,

            errors: [

                createError({

                    type:
                        "input",

                    message:
                        "Grounding Validator: Learning Analysis отсутствует"

                })

            ]

        };

    }


    /*
     * =====================================================
     * PROPOSED EXPERIENCE
     * =====================================================
     */


    const proposed =
        analysis.proposedExperience;


    if (
        !isObject(
            proposed
        )
    ) {

        return {

            valid:
                false,

            errors: [

                createError({

                    type:
                        "structure",

                    message:
                        "Grounding Validator: proposedExperience отсутствует"

                })

            ]

        };

    }


    /*
     * =====================================================
     * GROUNDING EVIDENCE
     * =====================================================
     */


    const groundingEvidence =
        Array.isArray(
            analysis.groundingEvidence
        )
            ? analysis.groundingEvidence
            : null;


    if (!groundingEvidence) {

        return {

            valid:
                false,

            errors: [

                createError({

                    type:
                        "structure",

                    message:
                        "Grounding Validator: groundingEvidence должен быть массивом"

                })

            ]

        };

    }


    /*
     * =====================================================
     * VALIDATE EVIDENCE STRUCTURE
     * =====================================================
     */


    for (
        const evidenceItem
        of groundingEvidence
    ) {


        if (
            !isObject(
                evidenceItem
            )
        ) {

            errors.push(

                createError({

                    type:
                        "structure",

                    message:
                        "Элемент groundingEvidence должен быть объектом"

                })

            );


            continue;

        }


        const field =
            String(
                evidenceItem.field || ""
            ).trim();


        const value =
            String(
                evidenceItem.value || ""
            ).trim();


        const evidence =
            String(
                evidenceItem.evidence || ""
            ).trim();


        /*
         * =================================================
         * FIELD
         * =================================================
         */


        if (
            !GROUNDED_FIELDS.includes(
                field
            )
        ) {

            errors.push(

                createError({

                    type:
                        "field",

                    field,

                    value,

                    evidence,

                    message:
                        `Недопустимое поле groundingEvidence: ${field || "(пусто)"}`

                })

            );


            continue;

        }


        /*
         * =================================================
         * VALUE
         * =================================================
         */


        if (!value) {

            errors.push(

                createError({

                    type:
                        "value",

                    field,

                    evidence,

                    message:
                        `groundingEvidence.${field}: value отсутствует`

                })

            );

        }


        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        if (!evidence) {

            errors.push(

                createError({

                    type:
                        "evidence",

                    field,

                    value,

                    message:
                        `groundingEvidence.${field}: evidence отсутствует`

                })

            );


            continue;

        }


        /*
         * =================================================
         * EVIDENCE MUST EXIST IN USER CORRECTION
         * =================================================
         *
         * Здесь модель уже не может
         * просто придумать цитату.
         *
         * После нормализации evidence
         * должен быть реальным фрагментом
         * correction пользователя.
         *
         * =================================================
         */


        const normalizedEvidence =
            normalizeText(
                evidence
            );


        if (
            !normalizedCorrection.includes(
                normalizedEvidence
            )
        ) {

            errors.push(

                createError({

                    type:
                        "unsupported_evidence",

                    field,

                    value,

                    evidence,

                    message:
                        "Evidence не найден в исходном correction пользователя"

                })

            );

        }

    }


    /*
     * =====================================================
     * CHECK EVERY PROPOSED RULE
     * =====================================================
     *
     * Для каждого элемента операционного поля
     * должна существовать ровно одна
     * соответствующая запись groundingEvidence.
     *
     * =====================================================
     */


    for (
        const field
        of GROUNDED_FIELDS
    ) {


        const values =
            Array.isArray(
                proposed[field]
            )
                ? proposed[field]
                : [];


        for (
            const proposedValue
            of values
        ) {


            const normalizedValue =
                normalizeText(
                    proposedValue
                );


            /*
             * Все evidence-записи,
             * относящиеся к этому правилу.
             */


            const matches =
                groundingEvidence.filter(
                    item => {


                        if (
                            !isObject(
                                item
                            )
                        ) {

                            return false;

                        }


                        return (
                            String(
                                item.field || ""
                            ).trim() ===
                            field
                            &&
                            normalizeText(
                                item.value
                            ) ===
                            normalizedValue
                        );

                    }
                );


            /*
             * =================================================
             * MISSING
             * =================================================
             */


            if (
                matches.length === 0
            ) {

                errors.push(

                    createError({

                        type:
                            "missing_evidence",

                        field,

                        value:
                            proposedValue,

                        message:
                            `Для proposedExperience.${field} отсутствует groundingEvidence`

                    })

                );


                continue;

            }


            /*
             * =================================================
             * DUPLICATE
             * =================================================
             */


            if (
                matches.length > 1
            ) {

                errors.push(

                    createError({

                        type:
                            "duplicate_evidence",

                        field,

                        value:
                            proposedValue,

                        message:
                            `Для proposedExperience.${field} найдено несколько groundingEvidence`

                    })

                );

            }

        }

    }


    /*
     * =====================================================
     * CHECK ORPHAN EVIDENCE
     * =====================================================
     *
     * Evidence не должен существовать
     * для правила, которого вообще нет
     * в proposedExperience.
     *
     * =====================================================
     */


    for (
        const evidenceItem
        of groundingEvidence
    ) {


        if (
            !isObject(
                evidenceItem
            )
        ) {

            continue;

        }


        const field =
            String(
                evidenceItem.field || ""
            ).trim();


        if (
            !GROUNDED_FIELDS.includes(
                field
            )
        ) {

            continue;

        }


        const proposedValues =
            Array.isArray(
                proposed[field]
            )
                ? proposed[field]
                : [];


        const normalizedEvidenceValue =
            normalizeText(
                evidenceItem.value
            );


        const exists =
            proposedValues.some(
                value =>
                    normalizeText(
                        value
                    ) ===
                    normalizedEvidenceValue
            );


        if (!exists) {

            errors.push(

                createError({

                    type:
                        "orphan_evidence",

                    field,

                    value:
                        evidenceItem.value,

                    evidence:
                        evidenceItem.evidence,

                    message:
                        "groundingEvidence относится к правилу, которого нет в proposedExperience"

                })

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

        checkedFields:
            [
                ...GROUNDED_FIELDS
            ],

        checkedEvidenceCount:
            groundingEvidence.length,

        errors

    };

}
