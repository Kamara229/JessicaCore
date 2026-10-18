/*
 * =========================================================
 * JESSICA SKILL QUALITY — EXECUTION PROGRAM
 * =========================================================
 */

function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


function normalizeTools(
    value
) {

    return new Set(

        Array.isArray(value)

            ? value
                .map(
                    item =>
                        normalizeText(
                            item
                        )
                )
                .filter(Boolean)

            : []

    );

}


export function inspectExecutionProgram(
    experience
) {

    const errors = [];
    const warnings = [];

    const program =
        experience?.executionProgram;


    if(!program){

        return {

            errors,

            warnings,

            stats: {
                hasExecutionProgram:
                    false
            }

        };

    }


    const strategy =

        normalizeText(
            program.strategy
        );


    const tools =

        normalizeTools(
            experience?.requiredTools
        );


    if(
        strategy ===
        "official_source_lookup"
    ){

        for(
            const required
            of [
                "web_search",
                "web_fetch"
            ]
        ){

            if(
                !tools.has(
                    required
                )
            ){

                errors.push(
                    "executionProgram official_source_lookup требует " + required
                );

            }

        }

    }


    if(
        strategy ===
        "date_difference"
        &&
        !tools.has(
            "date_diff"
        )
    ){

        errors.push(
            "executionProgram date_difference требует date_diff"
        );

    }


    if(
        !strategy
    ){

        errors.push(
            "executionProgram не содержит strategy"
        );

    }


    if(
        tools.size === 0
    ){

        warnings.push(
            "Executable Skill не содержит requiredTools"
        );

    }


    return {

        errors,

        warnings,

        stats: {

            hasExecutionProgram:
                true,

            strategy:

                strategy

                ||

                null,

            requiredTools:
                tools.size

        }

    };

}
