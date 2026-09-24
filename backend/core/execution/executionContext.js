/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT
 * =========================================================
 *
 * Контекст одной попытки выполнения.
 *
 * Используется:
 *
 * executionCycle
 * executionStepRunner
 * failureHandler
 *
 *
 * НЕ содержит:
 *
 * - выполнение;
 * - retry;
 * - replan;
 * - validation logic.
 *
 * Только хранит состояние.
 *
 * =========================================================
 */



export function createExecutionContext({

    task,

    plan,

    planningContext = {}

} = {}) {


    return {


        /*
         * исходная задача
         */

        task,



        /*
         * текущий план
         */

        plan,



        /*
         * контекст Experience / Planner
         */

        planningContext,



        /*
         * результаты этапов
         */

        runResult:
            null,


        answerResult:
            null,


        validationResult:
            null,



        /*
         * номер попытки
         */

        attempt:
            0,



        /*
         * история попыток
         */

        attempts:
            []

    };


}
