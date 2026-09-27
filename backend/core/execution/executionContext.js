/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT FACADE v2
 * =========================================================
 *
 * Единая точка доступа Execution Context.
 *
 *
 * Публичный API:
 *
 * - создание Context;
 * - изменение состояния;
 * - регистрация шагов;
 * - регистрация ошибок;
 * - counters;
 * - Retry;
 * - Replan.
 *
 *
 * Внутренние слои:
 *
 * contextFactory
 * contextReader
 * contextState
 * contextSteps
 * contextAttempts
 * contextFailures
 * contextRetry
 * contextReplan
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - принимает решения;
 * - управляет Execution Flow.
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

    updateExecutionState

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


    registerExecutionReplan

} from "./context/contextReplan.js";









/*
 * =========================================================
 * FINISH
 * =========================================================
 */


export {


    finishExecutionContext

} from "./context/contextState.js";
