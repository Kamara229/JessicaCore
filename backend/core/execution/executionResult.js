/*
 * =========================================================
 * JESSICA EXECUTION RESULT FACADE v3
 * =========================================================
 *
 * Единая точка доступа Execution Result слоя.
 *
 *
 * Ответственность:
 *
 * - экспортировать Result API;
 * - скрывать внутренние result modules;
 * - предоставлять единый контракт результата.
 *
 *
 * Внутренние слои:
 *
 * result/
 *
 * ├── resultStatus.js
 * ├── resultBuilders.js
 * ├── resultHelpers.js
 * └── resultMeta.js
 *
 *
 * НЕ:
 *
 * - создаёт Execution Context;
 * - анализирует Failure;
 * - принимает Terminal Decision;
 * - выполняет Execution.
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
 * META
 * =========================================================
 */


export {


    buildBaseResult,


    buildExecutionMeta,


    buildHistoryMeta,


    buildExperienceMeta,


    collectUsedTools



} from "./result/resultMeta.js";









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
