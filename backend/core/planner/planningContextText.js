import {
    normalizePlanningContext,
    hasPlanningContext
} from "./planningContext.js";


/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT
 * =========================================================
 *
 * Преобразует структурированный PlanningContext
 * в текст, понятный Planner.
 *
 * Этот модуль НЕ:
 *
 * - ищет Experience;
 * - сохраняет Experience;
 * - изменяет Planner;
 * - выполняет инструменты;
 * - принимает решения.
 *
 * Его задача только:
 *
 * PlanningContext
 *      ↓
 * текстовый блок для Planner
 *
 * =========================================================
 */


/*
 * =========================================================
 * FORMAT EXPERIENCE
 * =========================================================
 */


function formatExperience(
    experience
) {

    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return "";

    }


    const lines =
        [];


    /*
     * Основная информация о навыке.
     */


    if (experience.skillId) {

        lines.push(
            `Навык: ${experience.skillId}`
        );

    }


    if (experience.name) {

        lines.push(
            `Название: ${experience.name}`
        );

    }


    if (experience.version !== undefined) {

        lines.push(
            `Версия: ${experience.version}`
        );

    }


    if (experience.confidence !== undefined) {

        lines.push(
            `Уверенность: ${experience.confidence}`
        );

    }


    /*
     * Стратегия.
     */


    if (
        Array.isArray(
            experience.strategy
        ) &&
        experience.strategy.length > 0
    ) {

        lines.push(
            "",
            "Стратегия:"
        );


        experience.strategy.forEach(
            (
                item,
                index
            ) => {

                const text =
                    String(
                        item || ""
                    ).trim();


                if (text) {

                    lines.push(
                        `${index + 1}. ${text}`
                    );

                }

            }
        );

    }


    /*
     * Приоритет источников.
     */


    if (
        Array.isArray(
            experience.sourcePriority
        ) &&
        experience.sourcePriority.length > 0
    ) {

        lines.push(
            "",
            "Приоритет источников:"
        );


        experience.sourcePriority.forEach(
            (
                item,
                index
            ) => {

                const text =
                    String(
                        item || ""
                    ).trim();


                if (text) {

                    lines.push(
                        `${index + 1}. ${text}`
                    );

                }

            }
        );

    }


    /*
     * Правила проверки.
     */


    if (
        Array.isArray(
            experience.validationRules
        ) &&
        experience.validationRules.length > 0
    ) {

        lines.push(
            "",
            "Правила проверки:"
        );


        experience.validationRules.forEach(
            item => {

                const text =
                    String(
                        item || ""
                    ).trim();


                if (text) {

                    lines.push(
                        `- ${text}`
                    );

                }

            }
        );

    }


    /*
     * Известные ошибки.
     */


    if (
        Array.isArray(
            experience.failurePatterns
        ) &&
        experience.failurePatterns.length > 0
    ) {

        lines.push(
            "",
            "Известные ошибки, которых нужно избегать:"
        );


        experience.failurePatterns.forEach(
            item => {

                const text =
                    String(
                        item || ""
                    ).trim();


                if (text) {

                    lines.push(
                        `- ${text}`
                    );

                }

            }
        );

    }


    return lines
        .filter(
            item =>
                item !== null &&
                item !== undefined
        )
        .join(
            "\n"
        )
        .trim();

}


/*
 * =========================================================
 * FORMAT STRING LIST
 * =========================================================
 */


function formatList(
    title,
    items
) {

    if (
        !Array.isArray(items) ||
        items.length === 0
    ) {

        return "";

    }


    const lines = [
        title
    ];


    items.forEach(
        item => {

            const text =
                String(
                    item || ""
                ).trim();


            if (text) {

                lines.push(
                    `- ${text}`
                );

            }

        }
    );


    return lines
        .join(
            "\n"
        )
        .trim();

}


/*
 * =========================================================
 * BUILD PLANNING CONTEXT TEXT
 * =========================================================
 */


export function buildPlanningContextText(
    context
) {

    if (
        !hasPlanningContext(
            context
        )
    ) {

        return "";

    }


    const normalized =
        normalizePlanningContext(
            context
        );


    const sections =
        [];


    /*
     * EXPERIENCE
     */


    const experienceText =
        formatExperience(
            normalized.experience
        );


    if (experienceText) {

        sections.push(
            [
                "НАКОПЛЕННЫЙ ОПЫТ JESSICA:",
                experienceText
            ].join(
                "\n"
            )
        );

    }


    /*
     * SOURCE RULES
     */


    const sourceRulesText =
        formatList(
            "ПРАВИЛА РАБОТЫ С ИСТОЧНИКАМИ:",
            normalized.sourceRules
        );


    if (sourceRulesText) {

        sections.push(
            sourceRulesText
        );

    }


    /*
     * CONSTRAINTS
     */


    const constraintsText =
        formatList(
            "ОГРАНИЧЕНИЯ ЗАДАЧИ:",
            normalized.constraints
        );


    if (constraintsText) {

        sections.push(
            constraintsText
        );

    }


    /*
     * ADDITIONAL INSTRUCTIONS
     */


    const instructionsText =
        formatList(
            "ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ:",
            normalized.instructions
        );


    if (instructionsText) {

        sections.push(
            instructionsText
        );

    }


    /*
     * =====================================================
     * FINAL TEXT
     * =====================================================
     */


    if (
        sections.length === 0
    ) {

        return "";

    }


    return [
        "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ПЛАНИРОВАНИЯ:",
        "",
        sections.join(
            "\n\n"
        ),
        "",
        "Используй этот контекст как накопленный опыт и ограничения.",
        "Адаптируй его под текущую задачу, а не копируй механически."
    ].join(
        "\n"
    );

}
