/*
 * =========================================================
 * JESSICA TRACE TOOL EVIDENCE v1
 * =========================================================
 *
 * Извлекает из Execution Result только
 * ФАКТИЧЕСКИ выполненные инструменты.
 *
 *
 * Важно:
 *
 * Plan Step:
 *
 * {
 *   id,
 *   tool,
 *   arguments
 * }
 *
 * НЕ считается доказательством выполнения.
 *
 *
 * Tool Result:
 *
 * {
 *   tool,
 *   success,
 *   data / result / output / error
 * }
 *
 * считается фактическим Execution Evidence.
 *
 * =========================================================
 */


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


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
 * EXECUTION RESULT DETECTION
 * =========================================================
 */


function isExecutedToolResult(
    value
) {

    if(
        !isObject(value)
    ){

        return false;

    }


    const tool =

        normalizeText(
            value.tool
        );


    if(
        !tool
    ){

        return false;

    }


    /*
     * Самый сильный признак:
     * ToolRunner вернул explicit success.
     */


    if(
        typeof value.success === "boolean"
    ){

        return true;

    }


    /*
     * Compatibility для Tool Result,
     * если success отсутствует.
     *
     * Само наличие arguments НЕ является
     * доказательством выполнения:
     * это может быть просто Plan Step.
     */


    if(
        value.data !== undefined
        ||
        value.output !== undefined
        ||
        value.error !== undefined
        ||
        value.result !== undefined
    ){

        return true;

    }


    return false;

}


/*
 * =========================================================
 * RECURSIVE COLLECTOR
 * =========================================================
 */


function collectObjects(
    value,
    output,
    visited
) {

    if(
        value === null
        ||
        value === undefined
        ||
        typeof value !== "object"
    ){

        return;

    }


    if(
        visited.has(value)
    ){

        return;

    }


    visited.add(value);


    if(
        !Array.isArray(value)
    ){

        output.push(value);

    }


    const children =

        Array.isArray(value)

            ? value

            : Object.values(value);


    for(
        const child
        of children
    ){

        collectObjects(

            child,

            output,

            visited

        );

    }

}


/*
 * =========================================================
 * EXTRACT
 * =========================================================
 */


export function extractExecutedTools(
    executionResult
) {

    const objects = [];

    collectObjects(

        executionResult,

        objects,

        new Set()

    );


    const tools = [];


    for(
        const object
        of objects
    ){

        if(
            !isExecutedToolResult(
                object
            )
        ){

            continue;

        }


        const tool =

            normalizeText(
                object.tool
            );


        if(
            tool
            &&
            !tools.includes(tool)
        ){

            tools.push(tool);

        }

    }


    return tools;

}


/*
 * =========================================================
 * MERGE INTO TRACE
 * =========================================================
 */


export function collectTraceExecutedTools(

    trace,

    executionResult

) {

    if(
        !trace
    ){

        return trace;

    }


    if(
        !Array.isArray(
            trace.executedTools
        )
    ){

        trace.executedTools = [];

    }


    const tools =

        extractExecutedTools(
            executionResult
        );


    for(
        const tool
        of tools
    ){

        if(
            !trace.executedTools.includes(
                tool
            )
        ){

            trace.executedTools.push(
                tool
            );

        }

    }


    return trace;

}
