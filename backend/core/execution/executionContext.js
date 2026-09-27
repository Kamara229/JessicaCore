/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT FACADE v3
 * =========================================================
 *
 * Единая точка доступа Execution Context.
 *
 *
 * Ответственность:
 *
 * - предоставить публичный API Context слоя;
 * - скрыть внутреннюю структуру context modules.
 *
 *
 * Внутренние слои:
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
 * - управляет Retry;
 * - управляет Replan;
 * - меняет Execution Flow.
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


    getCurrentPlan,


    getInitialPlan,


    getExecutionState



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
