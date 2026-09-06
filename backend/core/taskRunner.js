import {
    executeTool,
    hasTool
} from "../tools/toolRegistry.js";


/*
 * =========================================================
 * JESSICA TASK RUNNER
 * =========================================================
 *
 * TaskRunner:
 *
 * 1. получает план от Planner;
 * 2. выполняет шаги последовательно;
 * 3. сохраняет результат каждого шага;
 * 4. позволяет следующим шагам использовать
 *    результаты предыдущих шагов.
 *
 *
 * Пример:
 *
 * {
 *   "id": "search",
 *   "tool": "web_search",
 *   "arguments": {
 *     "query": "NASA current missions"
 *   }
 * }
 *
 * затем:
 *
 * {
 *   "id": "page",
 *   "tool": "web_fetch",
 *   "arguments": {
 *     "url": {
 *       "$from": "search",
 *       "path": "data.results.0.url"
 *     }
 *   }
 * }
 *
 * TaskRunner сам подставит URL,
 * полученный от первого шага.
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
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
 *
 * Получает значение:
 *
 * data.results.0.url
 *
 * из объекта результата шага.
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
            .split(
                "."
            )
            .map(
                part =>
                    part.trim()
            )
            .filter(
                Boolean
            );


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
         * Защита от опасных prototype paths.
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


    return results.find(
        item =>
            item.id ===
            stepId
    ) || null;

}


/*
 * =========================================================
 * RESOLVE REFERENCE
 * =========================================================
 *
 * Поддерживаем:
 *
 * {
 *   "$from": "search",
 *   "path": "data.results.0.url"
 * }
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
            success: false,

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
            success: false,

            text:
                `Не найден результат шага ${from}`
        };

    }


    if (
        source.success !== true
    ) {

        return {
            success: false,

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
            success: false,

            text:
                (
                    `Не удалось получить ${path || "результат"} ` +
                    `из шага ${from}`
                )
        };

    }


    return {
        success: true,

        value
    };

}


/*
 * =========================================================
 * RESOLVE VALUE
 * =========================================================
 *
 * Рекурсивно обрабатывает аргументы.
 *
 * Это позволяет использовать ссылки
 * не только непосредственно в arguments.url,
 * но и внутри вложенных объектов и массивов.
 */


function resolveValue(
    value,
    results
) {

    /*
     * Простые значения.
     */
    if (
        value === null ||
        value === undefined ||
        typeof value !== "object"
    ) {

        return {
            success: true,

            value
        };

    }


    /*
     * Ссылка на предыдущий шаг.
     */
    if (
        !Array.isArray(
            value
        ) &&
        typeof value.$from === "string"
    ) {

        return resolveReference(
            value,
            results
        );

    }


    /*
     * Массив.
     */
    if (
        Array.isArray(
            value
        )
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
            success: true,

            value:
                resolvedArray
        };

    }


    /*
     * Обычный объект.
     */
    const resolvedObject =
        {};


    for (
        const [
            key,
            item
        ]
        of Object.entries(
            value
        )
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
        success: true,

        value:
            resolvedObject
    };

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


    /*
     * Старые планы без id
     * продолжают работать.
     */
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
            ids.has(
                id
            )
        ) {

            return {
                success: false,

                text:
                    `В плане повторяется id шага: ${id}`
            };

        }


        ids.add(
            id
        );

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * RUN PLAN
 * =========================================================
 */


export async function runPlan(
    plan
) {

    /*
     * -----------------------------------------------------
     * PLAN VALIDATION
     * -----------------------------------------------------
     */


    if (
        !plan ||
        typeof plan !== "object"
    ) {

        return {
            success: false,

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
            success: false,

            text:
                "В плане отсутствуют шаги",

            results:
                []
        };

    }


    /*
     * -----------------------------------------------------
     * NO TOOLS
     * -----------------------------------------------------
     */


    if (
        plan.requiresTools === false
    ) {

        return {
            success: true,

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
            success: false,

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
            success: false,

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
            success: false,

            text:
                idsValidation.text,

            results:
                []
        };

    }


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
                success: false,

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
                success: false,

                text:
                    `В шаге ${index + 1} отсутствует tool`,

                failedStep:
                    index,

                results
            };

        }


        /*
         * Runner повторно проверяет Registry,
         * даже если Planner уже сделал это.
         */
        if (
            !hasTool(
                toolName
            )
        ) {

            return {
                success: false,

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
         * -------------------------------------------------
         * RESOLVE ARGUMENTS
         * -------------------------------------------------
         *
         * Здесь происходит главное:
         * подстановка данных из предыдущих шагов.
         */


        const resolvedArgsResult =
            resolveValue(
                originalArgs,
                results
            );


        if (
            !resolvedArgsResult.success
        ) {

            return {
                success: false,

                stage:
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
         * -------------------------------------------------
         * EXECUTE TOOL
         * -------------------------------------------------
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
                success: false,

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
         * -------------------------------------------------
         * NEEDS CLARIFICATION
         * -------------------------------------------------
         */


        if (
            result.needsClarification
        ) {

            return {
                success: false,

                needsClarification:
                    true,

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
         * -------------------------------------------------
         * FAILED TOOL
         * -------------------------------------------------
         */


        if (
            !result.success
        ) {

            return {
                success: false,

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
        success: true,

        text:
            "План выполнен",

        results
    };

            }
