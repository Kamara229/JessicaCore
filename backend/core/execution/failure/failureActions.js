/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE ACTIONS v1
 * =========================================================
 *
 * Канонические действия Failure Decision.
 *
 *
 * Используется:
 *
 * - Failure Decision;
 * - Execution Failure Handler;
 * - Execution Loop.
 *
 *
 * НЕ:
 *
 * - принимает решения;
 * - анализирует Failure;
 * - изменяет Context;
 * - выполняет Retry;
 * - выполняет Replan.
 *
 * =========================================================
 */



/*
 * =========================================================
 * FAILURE ACTION
 * =========================================================
 */


export const FAILURE_ACTION = Object.freeze({

    RETRY:
        "RETRY",

    REPLAN:
        "REPLAN",

    CLARIFICATION:
        "CLARIFICATION",

    FINISH:
        "FINISH"

});
