/*
 * =========================================================
 * JESSICA EXECUTION RESULT FACADE v2
 * =========================================================
 *
 * Единая точка доступа Execution Result.
 *
 *
 * Публичный API:
 *
 * - статусы;
 * - создание результатов;
 * - проверки результатов.
 *
 *
 * Внутренние слои:
 *
 * resultMeta
 * resultBuilders
 * resultHelpers
 *
 *
 * НЕ:
 *
 * - содержит бизнес-логику;
 * - собирает metadata напрямую;
 * - анализирует ошибки.
 *
 * =========================================================
 */







/*
 * =========================================================
 * STATUS
 * =========================================================
 */


export {

    EXECUTION_RESULT_STATUS

} from "./result/resultStatus.js";









/*
 * =========================================================
 * BUILDERS
 * =========================================================
 */


export {


    buildCompletedResult,


    buildFailureResult,


    buildClarificationResult,


    buildNoVerifiedResult


} from "./result/resultBuilders.js";









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


export {


    isCompletedResult,


    isFailedResult,


    isClarificationResult,


    isNoVerifiedResult


} from "./result/resultHelpers.js";
