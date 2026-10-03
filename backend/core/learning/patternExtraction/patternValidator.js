/*
 * =========================================================
 * JESSICA PATTERN VALIDATOR v1
 * =========================================================
 *
 * Проверяет Dynamic Experience Pattern.
 *
 *
 * Проверки:
 *
 * - структура;
 * - обязательные поля;
 * - generalization;
 * - отсутствие конкретных данных
 *   единичного примера.
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - изменяет Pattern;
 * - сохраняет Experience.
 *
 * =========================================================
 */





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





/*
 * =========================================================
 * CONCRETE DATA
 * =========================================================
 */


const CONCRETE_PATTERNS = [

    /https?:\/\/[^\s]+/i,

    /\bwww\.[^\s]+/i,

    /\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b/i,

    /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/i,

    /\b(?:\+?\d[\d\s\-()]{8,}\d)\b/

];





function containsConcreteData(
    value
) {

    const text =

        normalizeText(
            value
        );



    if(
        !text
    ){

        return false;

    }



    return CONCRETE_PATTERNS.some(

        pattern =>
            pattern.test(
                text
            )

    );

}





/*
 * =========================================================
 * STRUCTURE
 * =========================================================
 */


function validateStructure(
    pattern
) {


    if(
        !pattern ||
        typeof pattern !== "object"
    ){

        return {


            valid:
                false,


            stage:
                "structure",


            reason:
                "Pattern отсутствует"

        };

    }



    if(
        !normalizeText(
            pattern.name
        )
    ){

        return {


            valid:
                false,


            stage:
                "structure",


            reason:
                "Pattern name отсутствует"

        };

    }



    if(
        !Array.isArray(
            pattern.workflow
        )
        ||
        pattern.workflow.length === 0
    ){

        return {


            valid:
                false,


            stage:
                "structure",


            reason:
                "Pattern workflow отсутствует"

        };

    }



    if(
        !Array.isArray(
            pattern.triggerPatterns
        )
        ||
        pattern.triggerPatterns.length === 0
    ){

        return {


            valid:
                false,


            stage:
                "structure",


            reason:
                "Pattern triggerPatterns отсутствуют"

        };

    }



    if(
        !Array.isArray(
            pattern.validationRules
        )
        ||
        pattern.validationRules.length === 0
    ){

        return {


            valid:
                false,


            stage:
                "structure",


            reason:
                "Pattern validationRules отсутствуют"

        };

    }



    return {

        valid:
            true

    };

}





/*
 * =========================================================
 * GENERALIZATION
 * =========================================================
 */


function validateGeneralization(
    pattern
) {


    const fields = [

        pattern?.name,

        pattern?.description,

        ...(Array.isArray(pattern?.triggerPatterns)
            ? pattern.triggerPatterns
            : []),

        ...(Array.isArray(pattern?.workflow)
            ? pattern.workflow
            : []),

        ...(Array.isArray(pattern?.validationRules)
            ? pattern.validationRules
            : []),

        ...(Array.isArray(pattern?.constraints)
            ? pattern.constraints
            : []),

        ...(Array.isArray(pattern?.successfulPatterns)
            ? pattern.successfulPatterns
            : []),

        ...(Array.isArray(pattern?.failurePatterns)
            ? pattern.failurePatterns
            : []),

        ...(Array.isArray(pattern?.avoidPatterns)
            ? pattern.avoidPatterns
            : [])

    ];



    const leakedFields =

        fields.filter(
            containsConcreteData
        );



    if(
        leakedFields.length > 0
    ){

        return {


            valid:
                false,


            stage:
                "generalization",


            reason:

                "Pattern содержит конкретные данные единичного Execution",


            leakedFields

        };

    }



    return {

        valid:
            true

    };

}





/*
 * =========================================================
 * VALIDATE PATTERN
 * =========================================================
 */


export function validatePattern(
    pattern
) {


    const structure =

        validateStructure(
            pattern
        );



    if(
        !structure.valid
    ){

        return structure;

    }



    const generalization =

        validateGeneralization(
            pattern
        );



    if(
        !generalization.valid
    ){

        return generalization;

    }



    return {


        valid:
            true,


        stage:
            "completed",


        structure:
            true,


        generalization:
            true,


        reason:
            "Dynamic Experience Pattern прошёл проверку"

    };

}
