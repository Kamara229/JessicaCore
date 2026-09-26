/*
 * =========================================================
 * JESSICA PLANNING CONTEXT TEXT
 * =========================================================
 *
 * Преобразует PlanningContext
 * в текст для AI Planner.
 *
 *
 * Flow:
 *
 * PlanningContext
 *        ↓
 * Experience Renderer
 *        ↓
 * Planner Prompt
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - изменяет Skills;
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

                typeof item === "string"

                    ? item.trim()

                    : JSON.stringify(item)

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
 * FORMAT SKILL
 * =========================================================
 */


function formatSkill(
    skill
) {


    if (
        !skill ||
        typeof skill !== "object"
    ) {

        return "";

    }



    const blocks = [];





    if (
        skill.id ||
        skill.name
    ) {


        blocks.push(

            [

                "Навык:",

                skill.name ||
                skill.id

            ]

            .join("\n")

        );

    }





    if (
        skill.workflow &&
        Array.isArray(
            skill.workflow
        )
    ) {


        blocks.push(

            [

                "Workflow:",

                formatArray(
                    skill.workflow
                )

            ]

            .join("\n")

        );

    }






    if (
        skill.constraints &&
        Array.isArray(
            skill.constraints
        )
    ) {


        blocks.push(

            [

                "Ограничения:",

                formatArray(
                    skill.constraints
                )

            ]

            .join("\n")

        );

    }





    if (
        skill.examples &&
        Array.isArray(
            skill.examples
        )
    ) {


        blocks.push(

            [

                "Примеры:",

                formatArray(
                    skill.examples
                )

            ]

            .join("\n")

        );

    }





    return blocks.join(

        "\n\n"

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




    const blocks = [];






    blocks.push(

        [

            "Источник:",

            experience.source ||
            "unknown"

        ]

        .join("\n")

    );






    if (
        experience.confidence !== undefined
    ) {


        blocks.push(

            `Confidence: ${experience.confidence}`

        );

    }







    const skills =

        Array.isArray(
            experience?.experience?.skills
        )

            ? experience.experience.skills

            : [];








    if (
        skills.length > 0
    ) {


        blocks.push(

            [

                "Используемые Skills:",


                skills

                    .map(

                        skill =>

                            formatSkill(
                                skill
                            )

                    )

                    .filter(Boolean)

                    .join(
                        "\n\n---\n\n"
                    )


            ]

            .join("\n\n")

        );

    }






    return blocks.join(

        "\n\n"

    );


}








/*
 * =========================================================
 * FORMAT SIMPLE SECTION
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


    if (
        !text
    ) {

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


            ]

            .join("\n\n")

        );

    }








    /*
     * RULES
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

                "=== META ===",

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

            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ПЛАНИРОВАНИЯ:",

            "",

            blocks.join("\n\n"),


            "",


            "Используй опыт Jessica как рекомендации.",

            "Проверяй применимость перед использованием."

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
