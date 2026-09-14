import {
    requestPlan
} from "./planner/plannerRequest.js";

import {
    parsePlan
} from "./planner/planParser.js";

import {
    normalizePlan
} from "./planner/planNormalizer.js";

import {
    validatePlan
} from "./planner/planValidator.js";

import {
    MAX_PLANNER_ATTEMPTS,
    sleep,
    isRetryablePlannerError,
    getPlannerRetryDelay
} from "./planner/plannerRetry.js";


/*
 * =========================================================
 * JESSICA PLANNER
 * =========================================================
 *
 * Главный координатор планирования.
 *
 * Сам Planner не содержит
 * внутреннюю реализацию отдельных этапов.
 *
 *
 * Рабочая цепочка:
 *
 * task
 *   ↓
 * PlanningContext
 *   ↓
 * plannerRequest
 *   ↓
 * planParser
 *   ↓
 * planNormalizer
 *   ↓
 * planValidator
 *   ↓
 * retry при необходимости
 *
 *
 * Детальная логика находится в:
 *
 * core/planner/
 *
 * plannerRequest.js
 * plannerPrompt.js
 * plannerTools.js
 * planningContext.js
 * planningContextText.js
 * planParser.js
 * planNormalizer.js
 * planValidator.js
 * plannerRetry.js
 *
 *
 * Этот файл отвечает только за:
 *
 * - проверку входной задачи;
 * - передачу PlanningContext;
 * - управление попытками Planner;
 * - последовательный запуск этапов;
 * - возврат результата.
 *
 *
 * PlanningContext в будущем может содержать:
 *
 * - Experience Jessica;
 * - успешные алгоритмы;
 * - правила источников;
 * - ограничения задачи;
 * - дополнительные инструкции;
 * - контекст Earnings.
 *
 *
 * Если PlanningContext не передан,
 * Planner работает как раньше.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(

    task,

    context = {}

) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    const cleanTask =
        String(
            task || ""
        ).trim();


    if (!cleanTask) {

        return {

            success:
                false,

            text:
                "Задача для Planner не указана"

        };

    }


    /*
     * =====================================================
     * CONFIGURATION
     * =====================================================
     */


    if (
        !process.env.GROQ_API_KEY
    ) {

        return {

            success:
                false,

            text:
                "GROQ_API_KEY не настроен"

        };

    }


    /*
     * =====================================================
     * ATTEMPTS
     * =====================================================
     */


    let lastError =
        "Не удалось построить план";


    for (
        let attempt = 1;
        attempt <= MAX_PLANNER_ATTEMPTS;
        attempt++
    ) {


        try {


            /*
             * =================================================
             * 1. REQUEST PLAN
             * =================================================
             *
             * Передаём:
             *
             * - задачу;
             * - ошибку предыдущей попытки;
             * - PlanningContext.
             *
             * =================================================
             */


            const rawText =
                await requestPlan(

                    cleanTask,

                    attempt > 1
                        ? lastError
                        : "",

                    context

                );


            /*
             * =================================================
             * 2. PARSE PLAN
             * =================================================
             */


            const rawPlan =
                parsePlan(
                    rawText
                );


            /*
             * =================================================
             * 3. NORMALIZE PLAN
             * =================================================
             */


            const plan =
                normalizePlan(
                    rawPlan
                );


            if (!plan) {

                throw new Error(
                    "Не удалось нормализовать план"
                );

            }


            /*
             * =================================================
             * 4. VALIDATE PLAN
             * =================================================
             */


            const validation =
                validatePlan(
                    plan
                );


            if (
                !validation.success
            ) {


                lastError =
                    validation.text ||
                    "План не прошёл проверку";


                console.warn(

                    `Planner validation failed ` +
                    `[${attempt}/${MAX_PLANNER_ATTEMPTS}]:`,

                    lastError

                );


                continue;

            }


            /*
             * =================================================
             * SUCCESS
             * =================================================
             */


            console.log(

                "Jessica plan:",

                JSON.stringify(
                    plan
                )

            );


            return {

                success:
                    true,

                plan

            };


        } catch (error) {


            /*
             * =================================================
             * ERROR
             * =================================================
             */


            lastError =
                error?.message ||
                "Неизвестная ошибка Planner";


            console.error(

                `Planner error ` +
                `[${attempt}/${MAX_PLANNER_ATTEMPTS}]:`,

                lastError

            );


            /*
             * =================================================
             * RETRY
             * =================================================
             */


            if (
                attempt < MAX_PLANNER_ATTEMPTS &&
                isRetryablePlannerError(
                    error
                )
            ) {


                await sleep(

                    getPlannerRetryDelay(
                        attempt
                    )

                );


            }


        }


    }


    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {

        success:
            false,

        text:
            `Planner не смог создать корректный план: ${lastError}`

    };


}


/*
 * =========================================================
 * BACKWARD-COMPATIBLE EXPORT
 * =========================================================
 *
 * Старый вариант:
 *
 * planTask(task)
 *
 * продолжает работать.
 *
 *
 * Новый вариант:
 *
 * planTask(
 *     task,
 *     context
 * )
 *
 * позволяет передавать PlanningContext.
 *
 * =========================================================
 */


export async function planTask(

    task,

    context = {}

) {


    const result =
        await createPlan(

            task,

            context

        );


    if (
        result.success
    ) {

        return result.plan;

    }


    throw new Error(
        result.text
    );


}
