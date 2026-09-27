/*
 * =========================================================
 * JESSICA EXECUTION CONTEXT FACADE v1
 * =========================================================
 *
 * Единая точка доступа Execution Context.
 *
 *
 * Внутренние модули:
 *
 * contextFactory
 * contextState
 * contextSteps
 * contextAttempts
 * contextFailures
 *
 *
 * Отвечает:
 *
 * - экспорт публичного API Context.
 *
 *
 * НЕ:
 *
 * - хранит реализацию;
 * - меняет Execution Flow;
 * - принимает решения;
 * - управляет Retry/Replan.
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

    registerExecutionAttempt,

    registerExecutionReplan

} from "./context/contextAttempts.js";









/*
 * =========================================================
 * FAILURES
 * =========================================================
 */


export {

    registerExecutionFailure

} from "./context/contextFailures.js";
