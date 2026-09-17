import {
    selectSource
} from "../sourceSelector.js";


/*
 * =========================================================
 * JESSICA TASK RUNNER
 * ARGUMENT RESOLVER
 * =========================================================
 *
 * Отвечает за подготовку arguments шага.
 *
 *
 * Возможности:
 *
 * - разрешает $from;
 * - читает вложенные path;
 * - работает с массивами и объектами;
 * - перед web_fetch запускает Source Selector;
 * - сообщает о semantic route failure.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - делает Replan;
 * - управляет execution loop.
 *
 * =========================================================
 */


/*
 * =========================================================
 * READ OBJECT PATH
 * =========================================================
 */


function getValueByPath(
    source,
    path
) {

    if (
        !source ||
        typeof source !== "object"
    ) {

        return undefined;

    }


    if (
        typeof path !== "string" ||
        !path.trim()
    ) {

        return source;

    }


    const parts =
        path
            .split(".")
            .map(
                item =>
                    item.trim()
            )
            .filter(Boolean);



    let current =
        source;



    for (
        const part
        of parts
    ) {


        if (
            current === null ||
            current === undefined
        ) {

            return undefined;

        }



        /*
         * Защита от prototype traversal.
         */


        if (
            part === "__proto__" ||
            part === "prototype" ||
            part === "constructor"
        ) {

            return undefined;

        }



        current =
            current[part];

    }



    return current;

}



/*
 * =========================================================
 * FIND STEP RESULT
 * =========================================================
 */


function findStepResult(
    stepId,
    results
) {

    if (
        typeof stepId !== "string" ||
        !stepId.trim()
    ) {

        return null;

    }


    return (

        results.find(
            item =>
                item?.id === stepId.trim()
        )

        || null

    );

}



/*
 * =========================================================
 * REFERENCE FAILURE
 * =========================================================
 */


function buildReferenceFailure(
    failureType,
    text
) {

    return {

        success:
            false,

        shouldRetry:
            false,

        stage:
            "argument-resolution",

        failureType,

        text

    };

}



/*
 * =========================================================
 * RESOLVE REFERENCE
 * =========================================================
 */


function resolveReference(
    reference,
    results
) {

    const from =
        typeof reference?.$from === "string"

            ? reference.$from.trim()

            : "";



    if (!from) {

        return buildReferenceFailure(

            "missing-from",

            "В ссылке на предыдущий шаг отсутствует $from"

        );

    }



    const source =
        findStepResult(
            from,
            results
        );



    if (!source) {

        return buildReferenceFailure(

            "previous-step-not-found",

            `Не найден результат шага ${from}`

        );

    }



    if (
        source.success !== true
    ) {

        return buildReferenceFailure(

            "previous-step-failed",

            `Шаг ${from} завершился неуспешно`

        );

    }



    const path =
        typeof reference?.path === "string"

            ? reference.path.trim()

            : "";



    const value =
        getValueByPath(
            source,
            path
        );



    if (
        value === undefined
    ) {

        return buildReferenceFailure(

            "reference-path-not-found",

            (
                `Не удалось получить ` +
                `${path || "результат"} ` +
                `из шага ${from}`
            )

        );

    }



    return {

        success:
            true,

        value

    };

}



/*
 * =========================================================
 * RESOLVE VALUE
 * =========================================================
 */


function resolveValue(
    value,
    results
) {

    if (
        value === null ||
        value === undefined ||
        typeof value !== "object"
    ) {

        return {

            success:
                true,

            value

        };

    }



    /*
     * $from reference
     */


    if (
        !Array.isArray(value) &&
        typeof value.$from === "string"
    ) {

        return resolveReference(
            value,
            results
        );

    }



    /*
     * ARRAY
     */


    if (
        Array.isArray(value)
    ) {

        const output =
            [];


        for (
            const item
            of value
        ) {

            const resolved =
                resolveValue(
                    item,
                    results
                );


            if (
                !resolved.success
            ) {

                return resolved;

            }


            output.push(
                resolved.value
            );

        }


        return {

            success:
                true,

            value:
                output

        };

    }



    /*
     * OBJECT
     */


    const output =
        {};


    for (
        const [key, item]
        of Object.entries(value)
    ) {

        const resolved =
            resolveValue(
                item,
                results
            );


        if (
            !resolved.success
        ) {

            return resolved;

        }


        output[key] =
            resolved.value;

    }



    return {

        success:
            true,

        value:
            output

    };

}



/*
 * =========================================================
 * FETCH SOURCE
 * =========================================================
 */


async function resolveFetchSource(
    originalArgs,
    results,
    selectionContext
) {

    const urlReference =
        originalArgs?.url;



    /*
     * Обычный URL.
     *
     * Source Selector нужен только тогда,
     * когда URL получаем из предыдущего web_search.
     */


    if (
        !urlReference ||
        typeof urlReference !== "object" ||
        Array.isArray(urlReference) ||
        typeof urlReference.$from !== "string"
    ) {

        return null;

    }



    const source =
        findStepResult(

            urlReference.$from,

            results

        );



    if (
        !source ||
        source.success !== true ||
        source.tool !== "web_search"
    ) {

        return null;

    }



    const searchResults =
        source?.data?.results;



    /*
     * Поиск не дал результатов.
     */


    if (
        !Array.isArray(
            searchResults
        ) ||
        searchResults.length === 0
    ) {

        return {

            success:
                false,

            shouldRetry:
                true,

            stage:
                "source-selection",

            failureType:
                "no-search-results",

            text:
                "Поиск не вернул подходящих источников"

        };

    }



    /*
     * =====================================================
     * SOURCE SELECTOR
     * =====================================================
     */


    let selection;


    try {


        selection =
            await selectSource(

                selectionContext,

                searchResults

            );


    } catch (error) {


        console.error(
            "Jessica Source Selector exception:",
            error
        );


        return {

            success:
                false,

            shouldRetry:
                false,

            stage:
                "source-selection",

            failureType:
                "source-selector-error",

            text:
                "Ошибка выбора источника"

        };

    }



    /*
     * =====================================================
     * NO SUITABLE SOURCE
     * =====================================================
     */


    if (
        selection?.noSuitableSource === true
    ) {

        return {

            success:
                false,

            shouldRetry:
                true,

            stage:
                "source-selection",

            failureType:
                "no-suitable-source",

            text:
                selection?.reason ||
                "Ни один найденный источник не соответствует требованиям задачи"

        };

    }



    /*
     * =====================================================
     * SELECTOR FAILURE
     * =====================================================
     */


    if (
        selection?.success !== true ||
        !selection?.result?.url
    ) {

        return {

            success:
                false,

            shouldRetry:
                false,

            stage:
                "source-selection",

            failureType:
                "source-selector-error",

            text:
                selection?.reason ||
                "Не удалось выбрать подходящий источник"

        };

    }



    return {

        success:
            true,

        url:
            selection.result.url

    };

}



/*
 * =========================================================
 * RESOLVE STEP ARGUMENTS
 * =========================================================
 */


export async function resolveStepArguments(

    toolName,

    originalArgs,

    results,

    selectionContext

) {


    /*
     * =====================================================
     * WEB FETCH
     * =====================================================
     */


    if (
        toolName === "web_fetch"
    ) {

        const selectedSource =
            await resolveFetchSource(

                originalArgs,

                results,

                selectionContext

            );



        if (selectedSource) {


            if (
                !selectedSource.success
            ) {

                return selectedSource;

            }



            /*
             * URL уже выбрал Source Selector.
             *
             * Остальные arguments всё равно
             * разрешаем стандартным $from.
             */


            const argsWithoutUrl = {

                ...(originalArgs || {})

            };


            delete argsWithoutUrl.url;



            const rest =
                resolveValue(

                    argsWithoutUrl,

                    results

                );



            if (
                !rest.success
            ) {

                return rest;

            }



            return {

                success:
                    true,

                value: {

                    ...rest.value,

                    url:
                        selectedSource.url

                }

            };

        }

    }



    /*
     * =====================================================
     * STANDARD RESOLUTION
     * =====================================================
     */


    return resolveValue(

        originalArgs,

        results

    );

}
