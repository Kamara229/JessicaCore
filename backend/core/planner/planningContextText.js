/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT v3
 * =========================================================
 *
 * Renderer PlanningContext для AI Planner.
 *
 *
 * PRIORITY:
 *
 * 1. instructions;
 * 2. constraints;
 * 3. Experience;
 * 4. plannerHints.
 *
 *
 * Это критично для Replan:
 *
 * retry / diversification instructions
 * не должны быть вытеснены длинным Experience.
 *
 * =========================================================
 */


import {
    normalizePlanningContext,
    hasPlanningContext
} from "./planningContext.js";


/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


const MAX_CONTEXT_LENGTH =
    4500;


const MAX_DEFAULT_ARRAY_ITEMS =
    10;


const MAX_INSTRUCTION_ITEMS =
    24;


const MAX_CONSTRAINT_ITEMS =
    16;


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
    value,
    maxItems = MAX_DEFAULT_ARRAY_ITEMS
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return "";

    }


    return value

        .slice(
            0,
            maxItems
        )

        .map(
            safeString
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
 * SECTION
 * =========================================================
 */


function formatSection(
    title,
    value,
    maxItems = MAX_DEFAULT_ARRAY_ITEMS
) {

    const text =

        formatArray(
            value,
            maxItems
        );


    if(
        !text
    ){

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

    if(
        !experience
        ||
        typeof experience !== "object"
    ){

        return "";

    }


    const blocks =
        [];


    /*
     * =====================================================
     * IDENTITY
     * =====================================================
     */


    const name =

        safeString(
            experience.name
        );


    if(
        name
    ){

        blocks.push(
            `Skill: ${name}`
        );

    }


    /*
     * =====================================================
     * DESCRIPTION
     * =====================================================
     */


    const description =

        safeString(
            experience.description
        );


    if(
        description
    ){

        blocks.push(

            [

                "Описание:",

                description

            ]
                .join("\n")

        );

    }


    /*
     * =====================================================
     * KNOWLEDGE
     * =====================================================
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

        .filter(Boolean)

        .join("\n\n");

}


/*
 * =========================================================
 * BUILD
 * =========================================================
 */


export function buildPlanningContextText(
    context
) {

    if(
        !hasPlanningContext(
            context
        )
    ){

        return "";

    }


    const normalized =

        normalizePlanningContext(
            context
        );


    const blocks =
        [];


    /*
     * =====================================================
     * 1. INSTRUCTIONS
     * =====================================================
     *
     * Ставим первыми.
     *
     * Retry / Replan constraints
     * должны пережить общий length limit.
     *
     * =====================================================
     */


    const instructions =

        formatSection(

            "=== ИНСТРУКЦИИ ===",

            normalized.instructions,

            MAX_INSTRUCTION_ITEMS

        );


    if(
        instructions
    ){

        blocks.push(
            instructions
        );

    }


    /*
     * =====================================================
     * 2. CONSTRAINTS
     * =====================================================
     */


    const constraints =

        formatSection(

            "=== ОГРАНИЧЕНИЯ ===",

            normalized.constraints,

            MAX_CONSTRAINT_ITEMS

        );


    if(
        constraints
    ){

        blocks.push(
            constraints
        );

    }


    /*
     * =====================================================
     * 3. EXPERIENCE
     * =====================================================
     */


    const experienceText =

        formatExperience(
            normalized.experience
        );


    if(
        experienceText
    ){

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
     * 4. PLANNER HINTS
     * =====================================================
     */


    const hints =

        formatSection(

            "=== PLANNER HINTS ===",

            normalized.plannerHints

        );


    if(
        hints
    ){

        blocks.push(
            hints
        );

    }


    /*
     * =====================================================
     * EMPTY
     * =====================================================
     */


    if(
        blocks.length === 0
    ){

        return "";

    }


    /*
     * =====================================================
     * FINAL
     * =====================================================
     */


    let result =

        [

            "КОНТЕКСТ JESSICA:",

            "",

            blocks.join(
                "\n\n"
            ),

            "",

            (
                "Experience является рекомендацией; " +
                "явные инструкции и ограничения текущей задачи имеют приоритет."
            )

        ]
            .join("\n")
            .trim();


    /*
     * =====================================================
     * SAFETY LIMIT
     * =====================================================
     */


    if(
        result.length
        >
        MAX_CONTEXT_LENGTH
    ){

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
