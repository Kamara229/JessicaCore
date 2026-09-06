import {
    executeTool,
    hasTool
} from "../tools/toolRegistry.js";

import {
    selectSource
} from "./sourceSelector.js";


/*
 * =========================================================
 * JESSICA TASK RUNNER
 * =========================================================
 *
 * TaskRunner:
 *
 * 1. получает план;
 * 2. выполняет шаги последовательно;
 * 3. сохраняет результаты;
 * 4. разрешает $from;
 * 5. перед web_fetch выбирает подходящий
 *    источник среди результатов web_search;
 * 6. может сообщить верхнему execution-циклу,
 *    что текущий маршрут нужно перестроить.
 *
 * ВАЖНО:
 *
 * TaskRunner сам НЕ делает replan.
 *
 * Он только возвращает:
 *
 * shouldRetry: true
 *
 * когда причина действительно допускает
 * альтернативный план.
 */


const MAX_STEPS =
    15;


/*
 * =========================================================
 * STEP RESULT NORMALIZATION
 * =========================================================
 */


function normalizeStepResult(
    step,
    result
) {

    return {
        id:
            step.id || null,

        tool:
            step.tool,

        arguments:
            step.arguments || {},

        success:
            result?.success === true,

        text:
            typeof result?.text === "string"
                ? result.text
                : "",

        data:
            result?.data ?? null,

        needsClarification:
            result?.needsClarification === true
    };

}


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
                part =>
                    part.trim()
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
 * FIND PREVIOUS RESULT
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
                item.id === stepId
        ) || null
    );

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

        return {
            success:
                false,

            stage:
                "argument-resolution",

            text:
                "В ссылке на предыдущий шаг отсутствует $from"
        };

    }


    const source =
        findStepResult(
            from,
            results
        );


    if (!source) {

        return {
            success:
                false,

            stage:
                "argument-resolution",

            text:
                `Не найден результат шага ${from}`
        };

    }


    if (
        source.success !== true
    ) {

        return {
            success:
                false,

            stage:
                "argument-resolution",

            text:
                `Шаг ${from} завершился неуспешно`
        };

    }


    const path =
        typeof reference.path === "string"
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

        return {
            success:
                false,

            stage:
                "argument-resolution",

            text:
                (
                    `Не удалось получить ${path || "результат"} ` +
                    `из шага ${from}`
                )
        };

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


    if (
        !Array.isArray(value) &&
        typeof value.$from === "string"
    ) {

        return resolveReference(
            value,
            results
        );

    }


    if (
        Array.isArray(value)
    ) {

        const resolvedArray =
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


            resolvedArray.push(
                resolved.value
            );

        }


        return {
            success:
                true,

            value:
                resolvedArray
        };

    }


    const resolvedObject =
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


        resolvedObject[key] =
            resolved.value;

    }


    return {
        success:
            true,

        value:
            resolvedObject
    };

}


/*
 * =========================================================
 * SOURCE SELECTION
 * =========================================================
 *
 * Если web_fetch получает URL из web_search,
 * Source Selector оценивает найденные варианты.
 *
 * Он может:
 *
 * SELECT
 * → вернуть URL;
 *
 * REJECT
 * → сообщить, что подходящего источника нет.
 *
 * REJECT является основанием для semantic retry,
 * но сам TaskRunner replan не выполняет.
 */


async function resolveFetchSource(
    originalArgs,
    results,
    selectionContext
) {

    const urlReference =
        originalArgs?.url;


    /*
     * URL не является ссылкой на web_search.
     *
     * Source Selector здесь не нужен.
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
            urlReference.$from.trim(),
            results
        );


    /*
     * Предыдущий шаг не является
     * успешным web_search.
     */
    if (
        !source ||
        source.success !== true ||
        source.tool !== "web_search"
    ) {

        return null;

    }


    const searchResults =
        source.data?.results;


    /*
     * Поиск отработал технически,
     * но не дал вариантов.
     *
     * Это не повод выполнять случайный fetch.
     * Нужен новый поисковый маршрут.
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


    const selection =
        await selectSource(
            selectionContext,
            searchResults
        );


    /*
     * =====================================================
     * SOURCE SELECTOR REJECTED ALL RESULTS
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
                selection.reason ||
                "Ни один найденный источник не соответствует требованиям задачи"
        };

    }


    /*
     * =====================================================
     * OTHER SELECTOR FAILURE
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


    /*
     * Source Selector сам пишет подробный лог
     * SELECT / REJECT.
     *
     * Здесь второй одинаковый лог больше
     * не создаём.
     */


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


async function resolveStepArguments(
    toolName,
    originalArgs,
    results,
    selectionContext
) {

    /*
     * =====================================================
     * WEB FETCH SOURCE SELECTION
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
             * Остальные arguments разрешаются
             * стандартным механизмом $from.
             *
             * URL заменяем результатом
             * Source Selector.
             */


            const argsWithoutUrl = {
                ...originalArgs
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
     * STANDARD ARGUMENT RESOLUTION
     * =====================================================
     */


    return resolveValue(
        originalArgs,
        results
    );

}


/*
 * =========================================================
 * NORMALIZE STEP ID
 * =========================================================
 */


function getStepId(
    step,
    index
) {

    if (
        typeof step?.id === "string" &&
        step.id.trim()
    ) {

        return step.id.trim();

    }


    return `step_${index + 1}`;

}


/*
 * =========================================================
 * VALIDATE UNIQUE IDS
 * =========================================================
 */


function validateStepIds(
    steps
) {

    const ids =
        new Set();


    for (
        let index = 0;
        index < steps.length;
        index++
    ) {

        const id =
            getStepId(
                steps[index],
                index
            );


        if (
            ids.has(id)
        ) {

            return {
                success:
                    false,

                text:
                    `В плане повторяется id шага: ${id}`
            };

        }


        ids.add(id);

    }


    return {
        success:
            true
    };

}


/*
 * =========================================================
 * RUN PLAN
 * =========================================================
 */


export async function runPlan(
    plan,
    task = ""
) {

    /*
     * =====================================================
     * PLAN VALIDATION
     * =====================================================
     */


    if (
        !plan ||
        typeof plan !== "object"
    ) {

        return {
            success:
                false,

            shouldRetry:
                false,

            text:
                "TaskRunner получил некорректный план",

            results:
                []
        };

    }


    if (
        !Array.isArray(
            plan.steps
        )
    ) {

        return {
            success:
                false,

            shouldRetry:
                false,

            text:
                "В плане отсутствуют шаги",

            results:
                []
        };

    }


    /*
     * =====================================================
     * NO TOOLS
     * =====================================================
     */


    if (
        plan.requiresTools === false
    ) {

        return {
            success:
                true,

            text:
                "Инструменты не требуются",

            results:
                []
        };

    }


    if (
        plan.steps.length === 0
    ) {

        return {
            success:
                false,

            shouldRetry:
                false,

            text:
                "План требует инструменты, но не содержит шагов",

            results:
                []
        };

    }


    if (
        plan.steps.length >
        MAX_STEPS
    ) {

        return {
            success:
                false,

            shouldRetry:
                false,

            text:
                (
                    `План содержит слишком много шагов: ` +
                    `${plan.steps.length}. Максимум: ${MAX_STEPS}.`
                ),

            results:
                []
        };

    }


    const idsValidation =
        validateStepIds(
            plan.steps
        );


    if (
        !idsValidation.success
    ) {

        return {
            success:
                false,

            shouldRetry:
                false,

            text:
                idsValidation.text,

            results:
                []
        };

    }


    /*
     * =====================================================
     * SOURCE SELECTION CONTEXT
     * =====================================================
     *
     * Предпочитаем исходную задачу.
     *
     * Для старых вызовов runPlan(plan)
     * сохраняется fallback на смысл плана.
     */


    const selectionContext =
        String(
            task || ""
        ).trim() ||
        [
            plan.intent || "",
            plan.reasoningSummary || "",
            plan.evidence?.reason || ""
        ]
            .filter(Boolean)
            .join("\n");


    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     */


    const results =
        [];


    for (
        let index = 0;
        index < plan.steps.length;
        index++
    ) {

        const originalStep =
            plan.steps[index];


        if (
            !originalStep ||
            typeof originalStep !== "object"
        ) {

            return {
                success:
                    false,

                shouldRetry:
                    false,

                text:
                    `Некорректный шаг ${index + 1}`,

                failedStep:
                    index,

                results
            };

        }


        const stepId =
            getStepId(
                originalStep,
                index
            );


        const toolName =
            typeof originalStep.tool === "string"
                ? originalStep.tool.trim()
                : "";


        if (!toolName) {

            return {
                success:
                    false,

                shouldRetry:
                    false,

                text:
                    `В шаге ${index + 1} отсутствует tool`,

                failedStep:
                    index,

                results
            };

        }


        if (
            !hasTool(
                toolName
            )
        ) {

            return {
                success:
                    false,

                shouldRetry:
                    false,

                text:
                    `Инструмент ${toolName} не зарегистрирован`,

                failedStep:
                    index,

                results
            };

        }


        const originalArgs =
            originalStep.arguments &&
            typeof originalStep.arguments === "object" &&
            !Array.isArray(
                originalStep.arguments
            )
                ? originalStep.arguments
                : {};


        /*
         * =================================================
         * RESOLVE ARGUMENTS
         * =================================================
         */


        let resolvedArgsResult;


        try {

            resolvedArgsResult =
                await resolveStepArguments(
                    toolName,
                    originalArgs,
                    results,
                    selectionContext
                );

        } catch (error) {

            console.error(
                `TaskRunner argument resolution exception [${stepId}]:`,
                error
            );


            return {
                success:
                    false,

                shouldRetry:
                    false,

                stage:
                    "argument-resolution",

                failureType:
                    "argument-resolution-error",

                text:
                    `Ошибка подготовки аргументов шага ${stepId}`,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results
            };

        }


        /*
         * =================================================
         * ARGUMENT / SOURCE FAILURE
         * =================================================
         */


        if (
            !resolvedArgsResult.success
        ) {

            console.warn(
                "Jessica TaskRunner route failure:",
                JSON.stringify({
                    stage:
                        resolvedArgsResult.stage ||
                        "argument-resolution",

                    failureType:
                        resolvedArgsResult.failureType ||
                        "argument-resolution",

                    shouldRetry:
                        resolvedArgsResult.shouldRetry === true,

                    reason:
                        resolvedArgsResult.text || ""
                })
            );


            return {
                success:
                    false,

                shouldRetry:
                    resolvedArgsResult.shouldRetry === true,

                stage:
                    resolvedArgsResult.stage ||
                    "argument-resolution",

                failureType:
                    resolvedArgsResult.failureType ||
                    "argument-resolution",

                text:
                    resolvedArgsResult.text ||
                    `Не удалось подготовить аргументы шага ${stepId}`,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results
            };

        }


        const resolvedArgs =
            resolvedArgsResult.value;


        console.log(
            (
                `Jessica TaskRunner: ` +
                `step ${index + 1}/${plan.steps.length} ` +
                `[${stepId}] -> ${toolName}`
            )
        );


        /*
         * =================================================
         * EXECUTE TOOL
         * =================================================
         */


        let rawResult;


        try {

            rawResult =
                await executeTool(
                    toolName,
                    resolvedArgs
                );

        } catch (error) {

            console.error(
                `TaskRunner tool exception [${toolName}]:`,
                error
            );


            return {
                success:
                    false,

                shouldRetry:
                    false,

                stage:
                    "tool",

                failureType:
                    "tool-exception",

                text:
                    `Ошибка выполнения инструмента ${toolName}`,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results
            };

        }


        const result =
            normalizeStepResult(
                {
                    id:
                        stepId,

                    tool:
                        toolName,

                    arguments:
                        resolvedArgs
                },

                rawResult
            );


        results.push(
            result
        );


        /*
         * =================================================
         * NEEDS CLARIFICATION
         * =================================================
         */


        if (
            result.needsClarification
        ) {

            return {
                success:
                    false,

                shouldRetry:
                    false,

                needsClarification:
                    true,

                stage:
                    "tool",

                text:
                    result.text ||
                    "Для выполнения задачи требуется уточнение.",

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results
            };

        }


        /*
         * =================================================
         * FAILED TOOL
         * =================================================
         */


        if (
            !result.success
        ) {

            return {
                success:
                    false,

                shouldRetry:
                    false,

                stage:
                    "tool",

                failureType:
                    "tool-failure",

                text:
                    result.text ||
                    `Не удалось выполнить шаг ${index + 1}`,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results
            };

        }

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {
        success:
            true,

        shouldRetry:
            false,

        text:
            "План выполнен",

        results
    };

}
