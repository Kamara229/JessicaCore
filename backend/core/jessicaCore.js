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
 * JESSICA CORE
 * =========================================================
 *
 * Главный цикл выполнения задачи:
 *
 * 1. Planner понимает задачу;
 * 2. TaskRunner выполняет инструменты;
 * 3. Answer Composer формирует ответ;
 * 4. Validator проверяет результат.
 *
 * В Jessica 4.0 сюда добавим:
 *
 * - повторное планирование;
 * - альтернативные стратегии;
 * - оценку качества;
 * - сохранение успешных маршрутов.
 */


/*
 * =========================================================
 * EXECUTE TASK
 * =========================================================
 */


export async function executeJessicaTask(
    task
) {

    /*
     * -----------------------------------------------------
     * 1. PLAN
     * -----------------------------------------------------
     */


    const planResult =
        await createPlan(
            task
        );


    if (
        !planResult.success
    ) {

        return {
            success: false,

            stage:
                "planner",

            text:
                planResult.text ||
                "Jessica не смогла понять задачу"
        };

    }


    const plan =
        planResult.plan;


    console.log(
        "Jessica plan:",
        JSON.stringify(
            plan
        )
    );


    /*
     * -----------------------------------------------------
     * 2. RUN PLAN
     * -----------------------------------------------------
     */


    const taskRunResult =
        await runPlan(
            plan
        );


    /*
     * Если инструмент прямо сообщил,
     * что необходимо уточнение пользователя,
     * не отправляем это дальше в Composer.
     */
    if (
        taskRunResult.needsClarification === true
    ) {

        return {
            success: false,

            needsClarification:
                true,

            stage:
                "tools",

            text:
                taskRunResult.text ||
                "Для выполнения задачи требуется уточнение.",

            plan,

            toolResults:
                taskRunResult.results || []
        };

    }


    /*
     * Если выполнение инструмента полностью
     * провалилось, пока останавливаем задачу.
     *
     * Позже Validator / Replanner сможет
     * попробовать другой маршрут.
     */
    if (
        taskRunResult.success !== true
    ) {

        return {
            success: false,

            stage:
                "tools",

            text:
                taskRunResult.text ||
                "Не удалось выполнить план Jessica",

            plan,

            toolResults:
                taskRunResult.results || []
        };

    }


    /*
     * -----------------------------------------------------
     * 3. COMPOSE ANSWER
     * -----------------------------------------------------
     */


    const answerResult =
        await composeAnswer(
            task,
            plan,
            taskRunResult
        );


    if (
        !answerResult.success
    ) {

        return {
            success: false,

            stage:
                "composer",

            text:
                answerResult.text ||
                "Jessica не смогла сформировать ответ",

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


    const validation =
        await validateResult(
            task,
            plan,
            taskRunResult,
            answerResult
        );


    /*
     * Validator сам сломался.
     *
     * Не выбрасываем уже хороший ответ,
     * но помечаем его как непроверенный.
     */
    if (
        !validation.success
    ) {

        return {
            success: true,

            text:
                answerResult.text,

            engine:
                "jessica-core",

            validated:
                false,

            validationReason:
                validation.reason ||
                "Validator недоступен",

            plan,

            toolResults:
                taskRunResult.results || []
        };

    }


    /*
     * Требуется уточнение.
     */
    if (
        validation.needsClarification === true
    ) {

        return {
            success: false,

            needsClarification:
                true,

            stage:
                "validator",

            text:
                validation.reason ||
                "Для выполнения задачи требуется уточнение.",

            plan,

            toolResults:
                taskRunResult.results || []
        };

    }


    /*
     * Validator считает результат плохим.
     *
     * В Jessica 1.0 мы только сообщаем об этом.
     *
     * В Jessica 4.0 здесь будет:
     *
     * Replanner
     *   ↓
     * новый план
     *   ↓
     * повторное выполнение
     */
    if (
        validation.valid !== true
    ) {

        return {
            success: false,

            shouldRetry:
                validation.shouldRetry === true,

            stage:
                "validator",

            text:
                (
                    "Jessica получила результат, " +
                    "но проверка качества показала, " +
                    "что задача решена недостаточно надёжно."
                ),

            validationReason:
                validation.reason || "",

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
        success: true,

        text:
            answerResult.text,

        engine:
            "jessica-core",

        answerSource:
            answerResult.source || "unknown",

        validated:
            true,

        validationReason:
            validation.reason || "",

        intent:
            plan.intent,

        usedTools:
            Array.isArray(
                taskRunResult.results
            )
                ? taskRunResult.results
                    .map(
                        result =>
                            result.tool
                    )
                : [],

        plan,

        toolResults:
            taskRunResult.results || []
    };

}
