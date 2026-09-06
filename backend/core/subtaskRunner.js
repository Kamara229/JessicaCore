import {
    createPlan
} from "./planner.js";

import {
    runPlan
} from "./taskRunner.js";

import {
    composeAnswer
} from "./answerComposer.js";

import {
    validateResult
} from "./validator.js";


/*
 * =========================================================
 * JESSICA SUBTASK RUNNER
 * =========================================================
 *
 * Выполняет одну подзадачу независимо.
 *
 * Главное отличие от общего Jessica Core:
 *
 * ошибка одной подзадачи НЕ должна
 * останавливать выполнение остальных.
 */


/*
 * =========================================================
 * EXECUTE ONE SUBTASK
 * =========================================================
 */


export async function executeSubtask(
    subtask
) {

    const subtaskId =
        subtask?.id ?? null;


    const taskText =
        typeof subtask?.text === "string"
            ? subtask.text.trim()
            : "";


    if (!taskText) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            result:
                "Подзадача не содержит текста"

        };

    }


    /*
     * -----------------------------------------------------
     * 1. PLAN
     * -----------------------------------------------------
     */


    let planResult;


    try {

        planResult =
            await createPlan(
                taskText
            );

    } catch (error) {

        console.error(
            `Subtask ${subtaskId} planner exception:`,
            error
        );


        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "planner",

            result:
                "Не удалось построить план"

        };

    }


    if (
        !planResult?.success
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "planner",

            result:
                planResult?.text ||
                "Не удалось построить план"

        };

    }


    const plan =
        planResult.plan;


    /*
     * -----------------------------------------------------
     * 2. RUN PLAN
     * -----------------------------------------------------
     */


    let taskRunResult;


    try {

        taskRunResult =
            await runPlan(
                plan
            );

    } catch (error) {

        console.error(
            `Subtask ${subtaskId} runner exception:`,
            error
        );


        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "runner",

            result:
                "Ошибка выполнения плана",

            plan

        };

    }


    /*
     * Нужна дополнительная информация от пользователя.
     */
    if (
        taskRunResult?.needsClarification === true
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "NEEDS_CLARIFICATION",

            success:
                false,

            needsClarification:
                true,

            result:
                taskRunResult.text ||
                "Для выполнения требуется уточнение",

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    /*
     * Ошибка выполнения инструмента.
     */
    if (
        taskRunResult?.success !== true
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "tools",

            result:
                taskRunResult?.text ||
                "Не удалось выполнить подзадачу",

            plan,

            toolResults:
                taskRunResult?.results || []

        };

    }


    /*
     * -----------------------------------------------------
     * 3. COMPOSE ANSWER
     * -----------------------------------------------------
     */


    let answerResult;


    try {

        answerResult =
            await composeAnswer(
                taskText,
                plan,
                taskRunResult
            );

    } catch (error) {

        console.error(
            `Subtask ${subtaskId} composer exception:`,
            error
        );


        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "composer",

            result:
                "Не удалось сформировать ответ",

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    if (
        !answerResult?.success
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            stage:
                "composer",

            result:
                answerResult?.text ||
                "Не удалось сформировать ответ",

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    /*
     * -----------------------------------------------------
     * 4. VALIDATE
     * -----------------------------------------------------
     */


    let validation;


    try {

        validation =
            await validateResult(
                taskText,
                plan,
                taskRunResult,
                answerResult
            );

    } catch (error) {

        console.error(
            `Subtask ${subtaskId} validator exception:`,
            error
        );


        /*
         * Validator не должен уничтожать
         * уже полученный хороший ответ.
         */
        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "COMPLETED",

            success:
                true,

            validated:
                false,

            result:
                answerResult.text,

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    if (
        validation?.needsClarification === true
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "NEEDS_CLARIFICATION",

            success:
                false,

            needsClarification:
                true,

            result:
                validation.reason ||
                "Для выполнения требуется уточнение",

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    /*
     * Validator считает результат ненадёжным.
     *
     * Пока не делаем retry автоматически.
     * Это появится позже через Replanner.
     */
    if (
        validation?.valid !== true
    ) {

        return {

            id:
                subtaskId,

            text:
                taskText,

            status:
                "FAILED",

            success:
                false,

            shouldRetry:
                validation?.shouldRetry === true,

            stage:
                "validator",

            result:
                validation?.reason ||
                "Результат не прошёл проверку качества",

            plan,

            toolResults:
                taskRunResult.results || []

        };

    }


    /*
     * -----------------------------------------------------
     * SUCCESS
     * -----------------------------------------------------
     */


    return {

        id:
            subtaskId,

        text:
            taskText,

        status:
            "COMPLETED",

        success:
            true,

        validated:
            true,

        result:
            answerResult.text,

        answerSource:
            answerResult.source || "unknown",

        usedTools:
            Array.isArray(
                taskRunResult.results
            )
                ? taskRunResult.results.map(
                    item =>
                        item.tool
                )
                : [],

        plan,

        toolResults:
            taskRunResult.results || []

    };

}


/*
 * =========================================================
 * RUN ALL SUBTASKS
 * =========================================================
 */


export async function runSubtasks(
    decomposition
) {

    const subtasks =
        Array.isArray(
            decomposition?.subtasks
        )
            ? decomposition.subtasks
            : [];


    if (
        subtasks.length === 0
    ) {

        return {

            success:
                false,

            text:
                "Нет подзадач для выполнения",

            results:
                []

        };

    }


    const results =
        [];


    /*
     * Пока выполняем последовательно.
     *
     * Это безопаснее для первой версии:
     * - меньше нагрузки на Groq;
     * - меньше запросов одновременно;
     * - проще отлаживать.
     *
     * Позже независимые задачи
     * можно будет запускать параллельно.
     */
    for (
        const subtask
        of subtasks
    ) {

        console.log(
            `Jessica subtask ${subtask.id}:`,
            subtask.text
        );


        try {

            const result =
                await executeSubtask(
                    subtask
                );


            results.push(
                result
            );

        } catch (error) {

            console.error(
                `Unhandled subtask error ${subtask.id}:`,
                error
            );


            results.push({

                id:
                    sub
