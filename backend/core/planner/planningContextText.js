/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT v2
 * =========================================================
 *
 * Компактный renderer PlanningContext для AI Planner.
 *
 * Важно:
 * PlanningContext не изменяется.
 *
 * Этот модуль только выбирает данные,
 * действительно необходимые Planner.
 *
 * =========================================================
 */


import {
    normalizePlanningContext,
    hasPlanningContext
} from "./planningContext.js";


const MAX_CONTEXT_LENGTH = 4500;

const MAX_ARRAY_ITEMS = 10;


/*
 * =========================================================
 * STRING
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


/*
 * =========================================================
 * ARRAY
 * =========================================================
 */


function formatArray(
    value
) {

    if (
        !Array.isArray(value)
    ) {

        return "";

    }


    return value
        .slice(
            0,
            MAX_ARRAY_ITEMS
        )
        .map(
            safeString
        )
        .filter(
            Boolean
        )
        .map(
            item =>
                `- ${item}`
        )
        .join("\n");

}


/*
 * =========================================================
 * SECTION
 * =========================================================
 */


function formatSection(
    title,
    value
) {

    const text =
        formatArray(
            value
        );


    if (!text) {

        return "";

    }


    return [
        title,
        text
    ]
        .join("\n");

}


/*
 * =========================================================
 * EXPERIENCE
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


    const blocks = [];


    /*
     * -----------------------------------------------------
     * IDENTITY
     * -----------------------------------------------------
     *
     * Planner достаточно понимать,
     * какой Skill применяется.
     *
     * Внутренние ID, version и confidence
     * остаются в PlanningContext,
     * но в AI prompt не передаются.
     */


    const name =
        safeString(
            experience.name
        );


    if (name) {

        blocks.push(
            `Skill: ${name}`
        );

    }


    /*
     * -----------------------------------------------------
     * DESCRIPTION
     * -----------------------------------------------------
     */


    const description =
        safeString(
            experience.description
        );


    if (description) {

        blocks.push(
            [
                "Описание:",
                description
            ]
                .join("\n")
        );

    }


    /*
     * -----------------------------------------------------
     * EXECUTION KNOWLEDGE
     * -----------------------------------------------------
     */


    blocks.push(
        formatSection(
            "Стратегия:",
            experience.strategy
        )
    );


    blocks.push(
        formatSection(
            "Приоритет источников:",
            experience.sourcePriority
        )
    );


    blocks.push(
        formatSection(
            "Правила проверки:",
            experience.validationRules
        )
    );


    blocks.push(
        formatSection(
            "Известные ошибки:",
            experience.failurePatterns
        )
    );


    blocks.push(
        formatSection(
            "Успешные подходы:",
            experience.successfulPatterns
        )
    );


    blocks.push(
        formatSection(
            "Избегать:",
            experience.avoidPatterns
        )
    );


    return blocks
        .filter(
            Boolean
        )
        .join("\n\n");

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
     * =====================================================
     * EXPERIENCE
     * =====================================================
     */


    const experienceText =
        formatExperience(
            normalized.experience
        );


    if (experienceText) {

        blocks.push(
            [
                "=== EXPERIENCE ===",
                experienceText
            ]
                .join("\n\n")
        );

    }


    /*
     * =====================================================
     * CONTEXT-LEVEL RULES
     * =====================================================
     *
     * sourceRules намеренно НЕ выводим отдельно.
     *
     * Они формируются из sourcePriority Experience
     * и иначе дублируют уже переданную информацию.
     * =====================================================
     */


    const constraints =
        formatSection(
            "=== ОГРАНИЧЕНИЯ ===",
            normalized.constraints
        );


    if (constraints) {

        blocks.push(
            constraints
        );

    }


    const instructions =
        formatSection(
            "=== ИНСТРУКЦИИ ===",
            normalized.instructions
        );


    if (instructions) {

        blocks.push(
            instructions
        );

    }


    const hints =
        formatSection(
            "=== PLANNER HINTS ===",
            normalized.plannerHints
        );


    if (hints) {

        blocks.push(
            hints
        );

    }


    /*
     * =====================================================
     * NO CONTEXT
     * =====================================================
     */


    if (
        blocks.length === 0
    ) {

        return "";

    }


    /*
     * =====================================================
     * FINAL TEXT
     * =====================================================
     */


    let result =
        [
            "КОНТЕКСТ JESSICA:",
            "",
            blocks.join("\n\n"),
            "",
            "Experience является рекомендацией; применяй только релевантные правила."
        ]
            .join("\n")
            .trim();


    /*
     * =====================================================
     * SAFETY LIMIT
     * =====================================================
     */


    if (
        result.length >
        MAX_CONTEXT_LENGTH
    ) {

        result =
            (
                result.slice(
                    0,
                    MAX_CONTEXT_LENGTH
                )
                +
                "\n\n[Контекст сокращён]"
            );

    }


    return result;

}
