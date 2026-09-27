/*
 * =========================================================
 * JESSICA EXECUTION RESULT CONTRACT v1
 * =========================================================
 *
 * Базовый контракт Execution Result.
 *
 * НЕ:
 *
 * - создаёт результаты;
 * - анализирует ошибки;
 * - собирает metadata.
 *
 * =========================================================
 */


export const EXECUTION_RESULT_STATUS = {


    COMPLETED:
        "COMPLETED",


    FAILED:
        "FAILED",


    NEEDS_CLARIFICATION:
        "NEEDS_CLARIFICATION",


    NO_VERIFIED_RESULT:
        "NO_VERIFIED_RESULT"


};
