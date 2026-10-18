/*
 * =========================================================
 * JESSICA DYNAMIC PATTERN QUALITY
 * =========================================================
 */

export function validateDynamicPatternQuality(
    proposal
) {

    if(
        proposal
            ?.provenance
            ?.dynamicPattern !== true
    ){

        return {

            passed:
                true,

            reason:
                "Dynamic Pattern не используется"

        };

    }


    const discovery =

        proposal
            ?.analysis
            ?.discovery;


    if(
        discovery?.resolved !== true
    ){

        return {

            passed:
                false,

            reason:
                "Dynamic Pattern Discovery не завершён"

        };

    }


    const validation =
        discovery?.validation;


    if(
        !validation
        ||
        validation.valid !== true
    ){

        return {

            passed:
                false,

            reason:
                "Dynamic Pattern не прошёл Validation"

        };

    }


    return {

        passed:
            true,

        reason:
            "Dynamic Pattern подтверждён"

    };

}
