import {
    executeTool,
    hasTool
} from "../tools/toolRegistry.js";


/*
 * =========================================================
 * JESSICA TASK RUNNER
 * =========================================================
 *
 * TaskRunner получает готовый план от Planner
 * и выполняет его шаги через Tool Registry.
 *
 * Он НЕ решает задачу сам.
 * Он НЕ выбирает инструменты.
 * Он только исполняет план.
 */


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
 * RUN PLAN
 * =========================================================
 */


export async function runPlan(
    plan
) {

    if (
        !plan ||
        typeof plan !== "object"
    ) {

        return {
            success: false,
            text:
                "TaskRunner получил некорректный план",
            results: []
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
            results: []
        };

    }


    /*
     * Если инструменты не нужны,
     * Runner ничего не выполняет.
     */
    if (
        plan.requiresTools === false
    ) {

        return {
            success: true,
            text:
                "Инструменты не требуются",
            results: []
        };

    }


    const results = [];


    /*
     * Пока выполняем шаги строго последовательно.
     *
     * Позже научим Jessica:
     * - передавать результаты между шагами;
     * - выполнять независимые шаги параллельно;
     * - повторять неудачный шаг;
     * - перестраивать план.
     */
    for (
        let index = 0;
        index < plan.steps.length;
        index++
    ) {

        const step =
            plan.steps[index];


        if (
            !step ||
            typeof step !== "object"
        ) {

            return {
                success: false,
                text:
                    `Некорректный шаг ${index + 1}`,
                results
            };

        }


        const toolName =
            typeof step.tool === "string"
                ? step.tool.trim()
                : "";


        if (!toolName) {

            return {
                success: false,
                text:
                    `В шаге ${index + 1} отсутствует tool`,
                results
            };

        }


        /*
         * Дополнительная защита.
         *
         * Planner уже проверяет инструменты,
         * но Runner тоже не доверяет входному плану.
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
                results
            };

        }


        const args =
            step.arguments &&
            typeof step.arguments === "object" &&
            !Array.isArray(
                step.arguments
            )
                ? step.arguments
                : {};


        console.log(
            `Jessica TaskRunner: step ${index + 1}/${plan.steps.length} -> ${toolName}`
        );


        const rawResult =
            await executeTool(
                toolName,
                args
            );


        const result =
            normalizeStepResult(
                {
                    ...step,
                    arguments:
                        args
                },
                rawResult
            );


        results.push(
            result
        );


        /*
         * Если инструмент требует уточнения,
         * дальнейшее выполнение бессмысленно.
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

                results
            };

        }


        /*
         * Первая версия Runner:
         * при ошибке шага останавливаем план.
         *
         * В Jessica 4.0 здесь появится
         * восстановление и альтернативная стратегия.
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

                results
            };

        }

    }


    return {
        success: true,

        text:
            "План выполнен",

        results
    };

}
