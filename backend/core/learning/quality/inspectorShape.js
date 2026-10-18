/*
 * =========================================================
 * JESSICA SKILL QUALITY — SHAPE
 * =========================================================
 */

import {
    SKILL_QUALITY_LIMITS
} from "./inspectorLimits.js";


const ARRAY_LIMITS = {

    examples:
        "maxExamples",

    workflow:
        "maxWorkflow",

    triggerPatterns:
        "maxTriggerPatterns",

    keywords:
        "maxKeywords",

    validationRules:
        "maxValidationRules",

    constraints:
        "maxConstraints",

    requiredTools:
        "maxRequiredTools"

};


function jsonBytes(
    value
) {

    try {

        return Buffer.byteLength(

            JSON.stringify(
                value
            ),

            "utf8"

        );

    }catch(error){

        return Number.MAX_SAFE_INTEGER;

    }

}


function duplicateCount(
    value
) {

    if(
        !Array.isArray(value)
    ){

        return 0;

    }


    const normalized =

        value
            .map(
                item =>
                    String(
                        item || ""
                    )
                        .trim()
                        .toLowerCase()
            )
            .filter(Boolean);


    return (

        normalized.length

        -

        new Set(
            normalized
        )
            .size

    );

}


export function inspectSkillShape(
    experience
) {

    const errors = [];
    const warnings = [];

    const payloadBytes =

        jsonBytes(
            experience
        );


    if(
        payloadBytes >
        SKILL_QUALITY_LIMITS.maxPayloadBytes
    ){

        errors.push(
            "Skill payload превышает лимит "
            + SKILL_QUALITY_LIMITS.maxPayloadBytes
            + " bytes"
        );

    }


    for(
        const [
            field,
            limitName
        ]
        of Object.entries(
            ARRAY_LIMITS
        )
    ){

        const value =

            Array.isArray(
                experience?.[field]
            )

                ? experience[field]

                : [];


        const limit =

            SKILL_QUALITY_LIMITS[
                limitName
            ];


        if(
            value.length > limit
        ){

            errors.push(
                field + " превышает лимит " + limit
            );

        }


        const duplicates =

            duplicateCount(
                value
            );


        if(
            duplicates > 0
        ){

            warnings.push(
                field + " содержит " + duplicates + " дубликат(а/ов)"
            );

        }

    }


    return {

        errors,

        warnings,

        stats: {

            payloadBytes,

            limits:
                SKILL_QUALITY_LIMITS

        }

    };

}
