/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT FACADE v5
 * =========================================================
 *
 * Единая публичная точка доступа к Execution Context.
 *
 *
 * Ответственность:
 *
 * - предоставить публичный API Context слоя;
 * - скрыть внутренние context modules;
 * - не допускать прямой зависимости Execution слоя
 *   от внутренней структуры Context.
 *
 *
 * Internal:
 *
 * context/
 *
 * ├── contextFactory.js
 * ├── contextReader.js
 * ├── contextState.js
 * ├── contextSteps.js
 * ├── contextAttempts.js
 * ├── contextFailures.js
 * ├── contextRetry.js
 * └── contextReplan.js
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - принимает Execution решения;
 * - делает Retry;
 * - делает Replan;
 * - управляет Execution Flow;
 * - хранит собственное состояние.
 *
 * =========================================================
 */









/*
 * =========================================================
 * FACTORY
 * =========================================================
 */


export {

    createExecutionContext

} from "./context/contextFactory.js";









/*
 * =========================================================
 * READER
 * =========================================================
 */


export {

    getExecutionId,

    getTraceId,

    getExecutionCounters,

    getExperience,

    getRunResults,

    getExecutionFailures,

    getExecutionReplans,

    getExecutionHistory,

    getPlans

} from "./context/contextReader.js";









/*
 * =========================================================
 * STATE
 * =========================================================
 */


export {

    updateExecutionState,

    finishExecutionContext

} from "./context/contextState.js";









/*
 * =========================================================
 * STEPS
 * =========================================================
 */


export {

    registerExecutionStep

} from "./context/contextSteps.js";









/*
 * =========================================================
 * ATTEMPTS
 * =========================================================
 */


export {

    registerExecutionAttempt

} from "./context/contextAttempts.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


export {

    registerExecutionFailure

} from "./context/contextFailures.js";









/*
 * =========================================================
 * RETRY
 * =========================================================
 */


export {

    registerExecutionRetry

} from "./context/contextRetry.js";









/*
 * =========================================================
 * REPLAN
 * =========================================================
 */


export {

    registerExecutionReplan,

    resetExecutionAfterReplan

} from "./context/contextReplan.js";
