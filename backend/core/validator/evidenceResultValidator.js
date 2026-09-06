/*
 * =========================================================
 * JESSICA EVIDENCE RESULT VALIDATOR
 * =========================================================
 *
 * Проверяет не то, что Planner ХОТЕЛ получить,
 * а то, что TaskRunner РЕАЛЬНО получил.
 *
 * evidence:
 *
 * none
 * search_results
 * source_content
 */


/*
 * =========================================================
 * COLLECT OBJECTS
 * =========================================================
 *
 * TaskRunner со временем может менять структуру результата.
 * Поэтому не привязываемся жёстко к results[0].
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


    if (!Array.isArray(value)) {
        output.push(value);
    }


    const children =
        Array.isArray(value)
            ? value
            : Object.values(value);


    for (const child of children) {

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
 * SUCCESSFUL TOOL RESULT
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
 * SEARCH EVIDENCE
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
                object.data?.results;


            return (
                Array.isArray(results) &&
                results.length > 0
            );

        }
    );
}


/*
 * =========================================================
 * SOURCE CONTENT EVIDENCE
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
                object.data?.content;


            return (
                typeof content === "string" &&
                content.trim().length > 0
            );

        }
    );
}


/*
 * =========================================================
 * PUBLIC VALIDATOR
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
     * Внешние доказательства
     * не требовались.
     */
    if (
        mode === "none"
    ) {

        return {
            valid: true,
            shouldRetry: false,
            reason:
                "Внешние доказательства не требуются"
        };

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

            return {
                valid: true,
                shouldRetry: false,
                reason:
                    "Получены реальные результаты web_search"
            };

        }


        return {
            valid: false,
            shouldRetry: true,
            reason:
                "Planner требовал search_results, но реальные результаты поиска не получены"
        };

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

            return {
                valid: true,
                shouldRetry: false,
                reason:
                    "Источник действительно был загружен и содержит данные"
            };

        }


        return {
            valid: false,
            shouldRetry: true,
            reason:
                "Planner требовал source_content, но содержимое источника фактически не получено"
        };

    }


    /*
     * Неизвестный режим нельзя
     * молча считать успешным.
     */
    return {
        valid: false,
        shouldRetry: true,
        reason:
            `Неизвестный evidence.mode: ${mode}`
    };

}
