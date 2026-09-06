import {
    createPlan
} from "./planner.js";

import {
    executePlanCycle
} from "./execution/executionCycle.js";


/*
 * =========================================================
 * JESSICA SUBTASK RUNNER
 * =========================================================
 *
 * Координатор выполнения подзадач.
 *
 * Здесь остаётся только:
 *
 * 1. подготовка подзадачи;
 * 2. создание первого плана;
 * 3. передача управления Execution Cycle;
 * 4. последовательный запуск нескольких подзадач;
 * 5. формирование общей статистики.
 *
 *
 * Детальный цикл:
 *
 * run
 * → compose
 * → validate
 * → replan
 * → retry
 *
 * находится в:
 *
 * core/execution/executionCycle.js
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


    /*
     * =====================================================
     * INPUT VALIDATION
     * =====================================================
     */


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

            stage:
                "input",

            result:
                "Подзадача не содержит текста"
        };

    }


    /*
     * =====================================================
     * 1. CREATE INITIAL PLAN
     * =====================================================
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
        !planResult?.success ||
        !planResult?.plan
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


    /*
     * =====================================================
     * 2. EXECUTION CYCLE
     * =====================================================
     *
     * Дальше весь цикл выполняется отдельно:
     *
     * plan
     * → run
     * → compose
     * → validate
     *
     * Если Validator требует повтор:
     *
     * → replan
     * → retry
     */


    let executionResult;


    try {

        executionResult =
            await executePlanCycle(
                taskText,
                planResult.plan
            );

    } catch (error) {

        console.error(
            `Subtask ${subtaskId} execution cycle exception:`,
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
                "execution",

            result:
                "Непредвиденная ошибка цикла выполнения",

            plan:
                planResult.plan
        };

    }


    /*
     * =====================================================
     * 3. NORMALIZE SUBTASK RESULT
     * =====================================================
     *
     * Execution Cycle уже возвращает:
     *
     * status
     * success
     * result
     * plan
     * toolResults
     * validated
     * usedTools
     * answerSource
     * attempt
     *
     * Здесь только добавляем идентификатор
     * и исходный текст подзадачи.
     */


    return {
        id:
            subtaskId,

        text:
            taskText,

        ...executionResult
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


    /*
     * =====================================================
     * NO SUBTASKS
     * =====================================================
     */


    if (
        subtasks.length === 0
    ) {

        return {
            success:
                false,

            text:
                "Нет подзадач для выполнения",

            total:
                0,

            completed:
                0,

            needsClarification:
                0,

            failed:
                0,

            results:
                []
        };

    }


    const results =
        [];


    /*
     * =====================================================
     * SEQUENTIAL EXECUTION
     * =====================================================
     *
     * Пока запускаем подзадачи последовательно.
     *
     * Причины:
     *
     * - меньше нагрузка на Groq;
     * - проще контролировать rate limits;
     * - проще читать логи;
     * - retry одной подзадачи не мешает другим.
     */


    for (
        const subtask
        of subtasks
    ) {

        console.log(
            `Jessica subtask ${subtask?.id}:`,
            subtask?.text
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
                `Unhandled subtask error ${subtask?.id}:`,
                error
            );


            /*
             * Ошибка одной подзадачи
             * не останавливает остальные.
             */


            results.push({
                id:
                    subtask?.id ?? null,

                text:
                    subtask?.text || "",

                status:
                    "FAILED",

                success:
                    false,

                stage:
                    "subtask",

                result:
                    "Непредвиденная ошибка выполнения подзадачи"
            });

        }

    }


    /*
     * =====================================================
     * SUMMARY
     * =====================================================
     */


    const completed =
        results.filter(
            item =>
                item.status ===
                "COMPLETED"
        ).length;


    const needsClarification =
        results.filter(
            item =>
                item.status ===
                "NEEDS_CLARIFICATION"
        ).length;


    const failed =
        results.filter(
            item =>
                item.status ===
                "FAILED"
        ).length;


    return {
        success:
            completed > 0,

        total:
            results.length,

        completed,

        needsClarification,

        failed,

        results
    };

        }
