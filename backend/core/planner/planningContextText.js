/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT
 * =========================================================
 *
 * Преобразует PlanningContext
 * в текст для Planner.
 *
 *
 * Flow:
 *
 * PlanningContext
 *        ↓
 * Context Renderer
 *        ↓
 * Planner Prompt
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - хранит Skills;
 * - принимает решения;
 * - вызывает AI.
 *
 * =========================================================
 */


import {
    normalizePlanningContext,
    hasPlanningContext
} from "./planningContext.js";



const MAX_CONTEXT_LENGTH =
    6000;



/*
 * =========================================================
 * SAFE VALUE
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
 * FORMAT LIST
 * =========================================================
 */


function formatList(
    value
) {


    if (
        !Array.isArray(value)
    ) {

        return "";

    }


    return value

        .map(
            item =>
                safeString(item)
        )

        .filter(Boolean)

        .map(
            item =>
                `- ${item}`
        )

        .join("\n");

}



/*
 * =========================================================
 * FORMAT OBJECT SECTION
 * =========================================================
 */


function formatSection(
    title,
    value
) {


    const text =
        formatList(
            value
        );


    if (!text) {

        return "";

    }


    return [

        title,

        text

    ].join("\n");

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
     * Identity
     */


    const identity = [];


    if (
        experience.skillId
    ) {

        identity.push(
            `Skill: ${experience.skillId}`
        );

    }


    if (
        experience.version !== undefined
    ) {

        identity.push(
            `Version: ${experience.version}`
        );

    }


    if (
        experience.confidence !== undefined
    ) {

        identity.push(
            `Confidence: ${experience.confidence}`
        );

    }



    if (
        identity.length
    ) {

        blocks.push(

            [
                "Информация о Skill:",

                ...identity

            ].join("\n")

        );

    }



    /*
     * Strategy
     */


    blocks.push(

        formatSection(
            "Рекомендуемая стратегия:",
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
            "Успешные паттерны:",
            experience.successfulPatterns
        )

    );



    return blocks
        .filter(Boolean)
        .join("\n\n");

}



/*
 * =========================================================
 * METADATA
 * =========================================================
 */


function formatMetadata(
    metadata
) {


    if (
        !metadata ||
        typeof metadata !== "object"
    ) {

        return "";

    }


    const blocks = [];



    if (
        metadata.retry
    ) {

        blocks.push(

            [
                "Информация о предыдущей попытке:",

                JSON.stringify(
                    metadata.retry
                )

            ].join("\n")

        );

    }



    return blocks.join("\n\n");

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


    const experience =
        formatExperience(
            normalized.experience
        );


    if (
        experience
    ) {

        blocks.push(

            [
                "=== ОПЫТ JESSICA ===",

                experience

            ].join("\n\n")

        );

    }



    /*
     * GLOBAL RULES
     */


    const sourceRules =
        formatSection(
            "=== ПРАВИЛА ИСТОЧНИКОВ ===",
            normalized.sourceRules
        );


    if (
        sourceRules
    ) {

        blocks.push(
            sourceRules
        );

    }



    /*
     * CONSTRAINTS
     */


    const constraints =
        formatSection(
            "=== ОГРАНИЧЕНИЯ ===",
            normalized.constraints
        );


    if (
        constraints
    ) {

        blocks.push(
            constraints
        );

    }



    /*
     * INSTRUCTIONS
     */


    const instructions =
        formatSection(
            "=== ИНСТРУКЦИИ ===",
            normalized.instructions
        );


    if (
        instructions
    ) {

        blocks.push(
            instructions
        );

    }



    /*
     * METADATA
     */


    const metadata =
        formatMetadata(
            normalized.metadata
        );


    if (
        metadata
    ) {

        blocks.push(

            [
                "=== СЛУЖЕБНЫЙ КОНТЕКСТ ===",

                metadata

            ].join("\n\n")

        );

    }




    if (
        blocks.length === 0
    ) {

        return "";

    }



    let result = [

        "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ПЛАНИРОВАНИЯ:",

        "",

        blocks.join("\n\n"),

        "",

        "Используй этот контекст как рекомендации.",

        "Адаптируй его под текущую задачу."

    ]
    .join("\n")
    .trim();



    /*
     * Ограничиваем размер.
     */


    if (
        result.length >
        MAX_CONTEXT_LENGTH
    ) {

        result =
            result.slice(
                0,
                MAX_CONTEXT_LENGTH
            )
            +
            "\n\n[Контекст сокращён]";

    }



    return result;

}
