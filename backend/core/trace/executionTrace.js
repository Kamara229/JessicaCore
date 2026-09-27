/*
 * =========================================================
 * JESSICA EXECUTION TRACE v8
 * =========================================================
 *
 * Центральный внутренний API Execution Trace.
 *
 *
 * Flow:
 *
 * Execution Modules
 *        ↓
 * executionTrace.js
 *        ↓
 *
 * Trace Factory
 * Trace Events
 * Trace Result
 * Trace Lifecycle
 * Trace Learning
 *
 *
 * Ответственность:
 *
 * - предоставить единый Trace API;
 * - скрыть внутреннюю структуру Trace модулей;
 * - сохранить стабильные импорты для Execution слоя.
 *
 *
 * НЕ:
 *
 * - содержит бизнес-логику Trace;
 * - выполняет Execution;
 * - принимает Failure Decision;
 * - изменяет Skills.
 *
 * =========================================================
 */



/*
 * =========================================================
 * FACTORY
 * =========================================================
 */


export {

    createExecutionTrace

} from "./traceFactory.js";



/*
 * =========================================================
 * EVENTS
 * =========================================================
 */


export {

    addTraceEvent,

    addTraceAttempt,

    addTraceReplan

} from "./traceEvents.js";



/*
 * =========================================================
 * RESULT
 * =========================================================
 */


export {

    updateTraceFromResult,

    updateTraceFromSummary

} from "./traceResult.js";



/*
 * =========================================================
 * LIFECYCLE
 * =========================================================
 */


export {

    finishExecutionTrace

} from "./traceLifecycle.js";



/*
 * =========================================================
 * LEARNING
 * =========================================================
 */


export {

    buildLearningPayload

} from "./traceLearning.js";
