/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT
 * =========================================================
 *
 * Renderer PlanningContext для AI Planner.
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
 * Ответственность:
 *
 * - превратить Context в текст;
 * - передать опыт Jessica Planner;
 * - сохранить структуру.
 *
 *
 * НЕ:
 *
 * - ищет Experience;
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





const MAX_CONTEXT_LENGTH =
    6000;


const MAX_ARRAY_ITEMS =
    15;







/*
 * =========================================================
 * STRING
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
            item =>
                safeString(item)
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
 * SIMPLE SECTION
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
 * EXPERIENCE FORMAT
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

            `Skill ID: ${experience.skillId}`

        );

    }



    if (
        experience.name
    ) {

        identity.push(

            `Название: ${experience.name}`

        );

    }



    if (
        experience.version
    ) {

        identity.push(

            `Версия: ${experience.version}`

        );

    }




    if (
        experience.skillConfidence !== undefined
    ) {

        identity.push(

            `Skill confidence: ${experience.skillConfidence}`

        );

    }




    if (
        experience.matchConfidence !== undefined
    ) {

        identity.push(

            `Match confidence: ${experience.matchConfidence}`

        );

    }




    if (
        identity.length
    ) {


        blocks.push(

            [

                "=== ИНФОРМАЦИЯ О SKILL ===",

                identity.join("\n")

            ]

            .join("\n")

        );

    }







    /*
     * Description
     */


    if (
        experience.description
    ) {


        blocks.push(

            [

                "Описание:",

                experience.description

            ]

            .join("\n")

        );

    }







    /*
     * Knowledge
     */


    blocks.push(

        formatSection(

            "Ключевые слова:",

            experience.keywords

        )

    );



    blocks.push(

        formatSection(

            "Теги:",

            experience.tags

        )

    );








    /*
     * Strategy
     */


    blocks.push(

        formatSection(

            "Стратегия выполнения:",

            experience.strategy

        )

    );



    blocks.push(

        formatSection(

            "Приоритет источников:",

            experience.sourcePriority

        )

    );








    /*
     * Quality
     */


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



    blocks.push(

        formatSection(

            "Избегать:",

            experience.avoidPatterns

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



    if (
        Object.keys(metadata).length === 0
    ) {

        return "";

    }



    return JSON.stringify(

        metadata,

        null,

        2

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

                "=== ОПЫТ JESSICA ===",

                experienceText

            ]

            .join("\n\n")

        );

    }







    /*
     * SOURCE RULES
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
     * PLANNER HINTS
     */


    const hints =

        formatSection(

            "=== ПОДСКАЗКИ PLANNER ===",

            normalized.plannerHints

        );



    if (
        hints
    ) {

        blocks.push(
            hints
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

            ]

            .join("\n\n")

        );

    }







    if (
        blocks.length === 0
    ) {

        return "";

    }








    let result =

        [

            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ JESSICA:",

            "",

            blocks.join("\n\n"),

            "",

            "Используй опыт Jessica как рекомендации.",

            "Проверяй применимость опыта к текущей задаче."

        ]

        .join("\n")

        .trim();








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
