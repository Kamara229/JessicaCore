/*
 * =========================================================
 * JESSICA TRACE TOOL EVIDENCE v2
 * =========================================================
 *
 * Определяет инструменты, которые
 * ФАКТИЧЕСКИ были выполнены Execution Layer.
 *
 *
 * Источники доказательства:
 *
 * 1. executionMeta.usedTools
 *
 *    Канонический источник текущего
 *    Execution Result.
 *
 *    Это список, сформированный Execution,
 *    а не Planner.
 *
 *
 * 2. Реальные вложенные Tool Result
 *
 *    {
 *      tool,
 *      success,
 *      data / result / output / error
 *    }
 *
 *    Используются как дополнительный
 *    compatibility / evidence source.
 *
 *
 * НЕ используем:
 *
 * - plan.steps;
 * - currentPlan.steps;
 * - initialPlan.steps;
 * - reasoning;
 * - AI declarations;
 * - experienceUsed внутри Plan.
 *
 *
 * Поэтому Planner не может самостоятельно
 * объявить инструмент выполненным.
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


function normalizeToolArray(
    value
) {

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    const result = [];


    for(
        const item
        of value
    ){

        const tool =

            normalizeText(
                item
            );


        if(
            !tool
        ){

            continue;

        }


        if(
            !result.includes(
                tool
            )
        ){

            result.push(
                tool
            );

        }

    }


    return result;

}


function mergeTools(
    ...collections
) {

    const result = [];


    for(
        const collection
        of collections
    ){

        const tools =

            normalizeToolArray(
                collection
            );


        for(
            const tool
            of tools
        ){

            if(
                !result.includes(
                    tool
                )
            ){

                result.push(
                    tool
                );

            }

        }

    }


    return result;

}


/*
 * =========================================================
 * CANONICAL EXECUTION META
 * =========================================================
 *
 * Текущий Execution Layer уже формирует:
 *
 * result.executionMeta.usedTools
 *
 * Например:
 *
 * {
 *   executionMeta: {
 *     usedTools: [
 *       "web_search"
 *     ]
 *   }
 * }
 *
 *
 * Это основной источник,
 * потому что список формируется после
 * реального TaskRunner Execution.
 *
 * =========================================================
 */


function extractExecutionMetaTools(
    executionResult
) {

    if(
        !isObject(
            executionResult
        )
    ){

        return [];

    }


    const direct =

        normalizeToolArray(

            executionResult
                ?.executionMeta
                ?.usedTools

        );


    /*
     * Compatibility:
     *
     * некоторые aggregated results
     * могут сами содержать children.
     *
     * Каждый Child Result обрабатывается
     * также отдельно traceResult.js,
     * но здесь поддерживаем вложенность
     * для устойчивости контракта.
     */


    const childResults =

        Array.isArray(
            executionResult?.results
        )

            ? executionResult.results

            : [];


    const nested = [];


    for(
        const child
        of childResults
    ){

        nested.push(

            ...extractExecutionMetaTools(
                child
            )

        );

    }


    return mergeTools(

        direct,

        nested

    );

}


/*
 * =========================================================
 * TOOL RESULT DETECTION
 * =========================================================
 */


function isExecutedToolResult(
    value
) {

    if(
        !isObject(
            value
        )
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
     * Plan Step обычно выглядит так:
     *
     * {
     *   id,
     *   tool,
     *   arguments
     * }
     *
     * Само наличие tool + arguments
     * НЕ является доказательством выполнения.
     */


    if(
        typeof value.success === "boolean"
    ){

        return true;

    }


    /*
     * Реальный Tool Result может
     * не иметь explicit success,
     * но иметь execution output.
     */


    if(
        value.data !== undefined
        ||
        value.output !== undefined
        ||
        value.result !== undefined
        ||
        value.error !== undefined
    ){

        return true;

    }


    return false;

}


/*
 * =========================================================
 * RECURSIVE OBJECT COLLECTION
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
        visited.has(
            value
        )
    ){

        return;

    }


    visited.add(
        value
    );


    if(
        !Array.isArray(
            value
        )
    ){

        output.push(
            value
        );

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
 * NESTED TOOL RESULTS
 * =========================================================
 */


function extractNestedToolResults(
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
            !tools.includes(
                tool
            )
        ){

            tools.push(
                tool
            );

        }

    }


    return tools;

}


/*
 * =========================================================
 * EXTRACT EXECUTED TOOLS
 * =========================================================
 */


export function extractExecutedTools(
    executionResult
) {

    if(
        !executionResult
    ){

        return [];

    }


    /*
     * Канонический источник.
     */


    const metaTools =

        extractExecutionMetaTools(
            executionResult
        );


    /*
     * Дополнительное фактическое evidence.
     */


    const resultTools =

        extractNestedToolResults(
            executionResult
        );


    return mergeTools(

        metaTools,

        resultTools

    );

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


    trace.executedTools =

        mergeTools(

            trace.executedTools,

            tools

        );


    return trace;

}
