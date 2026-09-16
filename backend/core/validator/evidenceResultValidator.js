/*
 * =========================================================
 * JESSICA EVIDENCE RESULT VALIDATOR
 * =========================================================
 *
 * Проверяет фактически полученные данные выполнения.
 *
 * Проверяет:
 *
 * plan.evidence.mode
 *        ↓
 * реальные результаты TaskRunner
 *
 *
 * НЕ:
 *
 * - проверяет смысл ответа;
 * - вызывает AI;
 * - проверяет claims;
 * - меняет план.
 *
 * =========================================================
 */


/*
 * =========================================================
 * COLLECT OBJECTS
 * =========================================================
 */


function collectObjects(
    value,
    output = [],
    visited = new Set()
) {

    if (
        value === null ||
        value === undefined ||
        typeof value !== "object"
    ) {

        return output;

    }


    if (
        visited.has(value)
    ) {

        return output;

    }


    visited.add(value);


    if (
        !Array.isArray(value)
    ) {

        output.push(value);

    }


    const children =
        Array.isArray(value)
            ? value
            : Object.values(value);


    for (
        const child
        of children
    ) {

        collectObjects(
            child,
            output,
            visited
        );

    }


    return output;

}



/*
 * =========================================================
 * TOOL RESULT CHECK
 * =========================================================
 */


function isSuccessfulToolResult(
    object,
    toolName
) {

    return (

        object &&

        object.tool === toolName &&

        object.success !== false

    );

}



/*
 * =========================================================
 * SEARCH RESULT
 * =========================================================
 */


function hasSearchEvidence(
    objects
) {

    return objects.some(
        object => {


            if (
                !isSuccessfulToolResult(
                    object,
                    "web_search"
                )
            ) {

                return false;

            }


            const results =
                object?.data?.results;


            return (

                Array.isArray(results) &&

                results.length > 0

            );

        }
    );

}



/*
 * =========================================================
 * SOURCE CONTENT
 * =========================================================
 */


function hasSourceContentEvidence(
    objects
) {

    return objects.some(
        object => {


            if (
                !isSuccessfulToolResult(
                    object,
                    "web_fetch"
                )
            ) {

                return false;

            }


            const content =
                object?.data?.content;


            const url =
                object?.data?.url;


            return (

                typeof content === "string" &&

                content.trim().length > 0 &&

                (
                    !url ||
                    typeof url === "string"
                )

            );

        }
    );

}



/*
 * =========================================================
 * RESULT BUILDER
 * =========================================================
 */


function buildResult(
    valid,
    reason,
    shouldRetry = false
) {

    return {

        valid,

        shouldRetry,

        needsClarification:
            false,

        reason

    };

}



/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function validateEvidenceResult(

    plan,

    taskRunResult

) {


    const mode =
        plan?.evidence?.mode ||
        "none";



    /*
     * =====================================================
     * NONE
     * =====================================================
     */


    if (
        mode === "none"
    ) {

        return buildResult(
            true,
            "Внешние доказательства не требуются"
        );

    }



    const objects =
        collectObjects(
            taskRunResult
        );



    /*
     * =====================================================
     * SEARCH RESULTS
     * =====================================================
     */


    if (
        mode === "search_results"
    ) {


        if (
            hasSearchEvidence(
                objects
            )
        ) {

            return buildResult(
                true,
                "Получены реальные результаты web_search"
            );

        }


        return buildResult(
            false,
            "Planner требовал search_results, но результаты поиска отсутствуют",
            true
        );

    }



    /*
     * =====================================================
     * SOURCE CONTENT
     * =====================================================
     */


    if (
        mode === "source_content"
    ) {


        if (
            hasSourceContentEvidence(
                objects
            )
        ) {

            return buildResult(
                true,
                "Источник успешно загружен"
            );

        }


        return buildResult(
            false,
            "Planner требовал source_content, но содержимое источника не получено",
            true
        );

    }



    /*
     * =====================================================
     * UNKNOWN
     * =====================================================
     */


    return buildResult(
        false,
        `Неизвестный evidence.mode: ${mode}`,
        true
    );

}
