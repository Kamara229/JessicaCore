/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT
 * =========================================================
 *
 * Преобразует Planning Context
 * в текстовый блок для Planner.
 *
 *
 * Ответственность:
 *
 * PlanningContext
 *        ↓
 * Human/AI readable context
 *        ↓
 * Planner
 *
 *
 * Этот модуль НЕ:
 *
 * - ищет Experience;
 * - читает Storage;
 * - изменяет Skills;
 * - вызывает AI;
 * - принимает решения.
 *
 * =========================================================
 */


import {
    normalizePlanningContext,
    hasPlanningContext
} from "./planningContext.js";



/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


/*
 * =========================================================
 * FORMAT ARRAY
 * =========================================================
 */


function formatArray(
    items
) {


    if (
        !Array.isArray(items) ||
        items.length === 0
    ) {

        return "";

    }


    return items
        .map(
            item =>
                safeString(
                    item
                )
        )
        .filter(
            Boolean
        )
        .map(
            item =>
                `- ${item}`
        )
        .join(
            "\n"
        );

}



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


    const sections = [];



    /*
     * BASIC
     */


    const basic = [];


    if (
        experience.skillId
    ) {

        basic.push(
            `Skill ID: ${experience.skillId}`
        );

    }


    if (
        experience.name
    ) {

        basic.push(
            `Название: ${experience.name}`
        );

    }


    if (
        experience.version !== undefined
    ) {

        basic.push(
            `Версия: ${experience.version}`
        );

    }


    if (
        experience.confidence !== undefined
    ) {

        basic.push(
            `Уверенность Skill: ${experience.confidence}`
        );

    }


    if (
        basic.length > 0
    ) {

        sections.push(
            [
                "Информация о навыке:",
                ...basic
            ]
            .join(
                "\n"
            )
        );

    }



    /*
     * STRATEGY
     */


    const strategy =
        formatArray(
            experience.strategy
        );


    if (
        strategy
    ) {

        sections.push(
            [
                "Стратегия выполнения:",
                strategy
            ]
            .join(
                "\n"
            )
        );

    }



    /*
     * SOURCES
     */


    const sources =
        formatArray(
            experience.sourcePriority
        );


    if (
        sources
    ) {

        sections.push(
            [
                "Приоритет источников:",
                sources
            ]
            .join(
                "\n"
            )
        );

    }



    /*
     * VALIDATION
     */


    const validation =
        formatArray(
            experience.validationRules
        );


    if (
        validation
    ) {

        sections.push(
            [
                "Правила проверки:",
                validation
            ]
            .join(
                "\n"
            )
        );

    }



    /*
     * FAILURES
     */


    const failures =
        formatArray(
            experience.failurePatterns
        );


    if (
        failures
    ) {

        sections.push(
            [
                "Ошибки, которых нужно избегать:",
                failures
            ]
            .join(
                "\n"
            )
        );

    }



    return sections.join(
        "\n\n"
    );

}



/*
 * =========================================================
 * BUILD CONTEXT TEXT
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


    const blocks = [];



    /*
     * EXPERIENCE
     */


    const experienceText =
        formatExperience(
            normalized.experience
        );


    if (
        experienceText
    ) {

        blocks.push(
            [
                "=== НАКОПЛЕННЫЙ ОПЫТ JESSICA ===",
                experienceText
            ]
            .join(
                "\n\n"
            )
        );

    }



    /*
     * GLOBAL RULES
     */


    const sourceRules =
        formatArray(
            normalized.sourceRules
        );


    if (
        sourceRules
    ) {

        blocks.push(
            [
                "=== ПРАВИЛА ИСТОЧНИКОВ ===",
                sourceRules
            ]
            .join(
                "\n\n"
            )
        );

    }



    /*
     * CONSTRAINTS
     */


    const constraints =
        formatArray(
            normalized.constraints
        );


    if (
        constraints
    ) {

        blocks.push(
            [
                "=== ОГРАНИЧЕНИЯ ===",
                constraints
            ]
            .join(
                "\n\n"
            )
        );

    }



    /*
     * USER INSTRUCTIONS
     */


    const instructions =
        formatArray(
            normalized.instructions
        );


    if (
        instructions
    ) {

        blocks.push(
            [
                "=== ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ===",
                instructions
            ]
            .join(
                "\n\n"
            )
        );

    }



    if (
        blocks.length === 0
    ) {

        return "";

    }



    return [

        "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДЛЯ ПЛАНИРОВАНИЯ:",

        "",

        blocks.join(
            "\n\n"
        ),

        "",

        "Используй этот опыт как рекомендации.",
        "Не копируй его механически — адаптируй под текущую задачу."

    ]
    .join(
        "\n"
    )
    .trim();


}
