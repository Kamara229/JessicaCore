/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT FACADE v6
 * =========================================================
 *
 * Единая публичная точка доступа к Execution Context.
 *
 *
 * Отвечает:
 *
 * - предоставляет публичный API Context слоя;
 * - скрывает внутренние context modules;
 * - изолирует Execution слой от внутренней
 *   структуры Execution Context.
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
 * - определяет Retry;
 * - определяет Replan;
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

    getExecutionState,

    getExecutionCounters,

    getExperience,

    getRunResults,

    getTerminalResult,

    getResultHistory,

    getExecutionFailures,

    getExecutionAttempts,

    getExecutionReplans,

    getExecutionHistory,

    getInitialPlan,

    getCurrentPlan,

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

    registerExecutionStep,

    registerCompletedStep,

    registerFailedStep

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
 * FAILURES
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
