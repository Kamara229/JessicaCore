/*
 * =========================================================
 * JESSICA SKILL KNOWLEDGE FINGERPRINT
 * =========================================================
 *
 * Fingerprint включает только исполняемое / семантическое
 * знание Skill.
 *
 * Не включает:
 * - examples;
 * - learning metrics;
 * - runtime statistics;
 * - timestamps / version.
 *
 * Поэтому новый successful Example сам по себе
 * не считается новым знанием.
 * =========================================================
 */


import {
    createHash
} from "node:crypto";


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
        .trim();

}


function normalizeList(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    const result = [];

    const seen =
        new Set();


    for(
        const item
        of value
    ){

        const text =
            normalizeText(
                item
            );


        if(
            !text
            ||
            seen.has(
                text
            )
        ){

            continue;

        }


        seen.add(
            text
        );


        result.push(
            text
        );

    }


    return result;

}


function normalizeProgram(
    value
) {

    if(
        !value
        ||
        typeof value !== "object"
        ||
        Array.isArray(value)
    ){

        return null;

    }


    return {

        version:

            Number(
                value.version || 0
            ),

        mode:

            normalizeText(
                value.mode
            )
                .toLowerCase(),

        strategy:

            normalizeText(
                value.strategy
            )
                .toLowerCase(),

        evidenceMode:

            normalizeText(
                value.evidenceMode
            )
                .toLowerCase()

    };

}


export function buildSkillKnowledgeSnapshot(
    skill
) {

    const source =

        (
            skill
            &&
            typeof skill === "object"
            &&
            !Array.isArray(skill)
        )

            ? skill

            : {};


    return {

        name:
            normalizeText(
                source.name
            ),

        category:
            normalizeText(
                source.category
            ),

        description:
            normalizeText(
                source.description
            ),

        workflow:
            normalizeList(
                source.workflow
            ),

        triggerPatterns:
            normalizeList(
                source.triggerPatterns
            ),

        keywords:
            normalizeList(
                source.keywords
            ),

        validationRules:
            normalizeList(
                source.validationRules
            ),

        constraints:
            normalizeList(
                source.constraints
            ),

        strategy:
            normalizeList(
                source.strategy
            ),

        sourcePriority:
            normalizeList(
                source.sourcePriority
            ),

        requiredTools:
            normalizeList(
                source.requiredTools
            ),

        successfulPatterns:
            normalizeList(
                source.successfulPatterns
            ),

        failurePatterns:
            normalizeList(
                source.failurePatterns
            ),

        avoidPatterns:
            normalizeList(
                source.avoidPatterns
            ),

        executionProgram:
            normalizeProgram(
                source.executionProgram
            )

    };

}


export function fingerprintSkillKnowledge(
    skill
) {

    const snapshot =

        buildSkillKnowledgeSnapshot(
            skill
        );


    return createHash(
        "sha256"
    )
        .update(
            JSON.stringify(
                snapshot
            )
        )
        .digest(
            "hex"
        );

}
