/*
 * =========================================================
 * JESSICA SUBTASK RUNNER
 * =========================================================
 *
 * Центральная точка доступа
 * к системе выполнения подзадач.
 *
 *
 * Архитектура:
 *
 * Jessica Core
 *      ↓
 * subtaskRunner.js
 *      ↓
 * ┌─────────────────────────────┐
 * │ executeSubtask.js           │
 * │ runSubtasks.js              │
 * │ subtaskSummary.js           │
 * └─────────────────────────────┘
 *
 *
 * Этот файл НЕ содержит:
 *
 * - Planner;
 * - Experience;
 * - Execution Cycle;
 * - циклы выполнения;
 * - подсчёт статистики;
 * - Tool Registry;
 * - бизнес-логику.
 *
 *
 * Его задача:
 *
 * сохранить единый публичный интерфейс
 * для остальных частей Jessica.
 *
 * Благодаря этому существующий код:
 *
 * import {
 *     executeSubtask,
 *     runSubtasks
 * } from "./subtaskRunner.js";
 *
 * продолжает работать без изменений.
 *
 * =========================================================
 */


/*
 * =========================================================
 * EXECUTE ONE SUBTASK
 * =========================================================
 *
 * Реализация находится:
 *
 * core/subtask/executeSubtask.js
 *
 * Внутренняя цепочка:
 *
 * Subtask
 *   ↓
 * Experience
 *   ↓
 * PlanningContext
 *   ↓
 * Planner
 *   ↓
 * Execution Cycle
 *
 * =========================================================
 */


export {
    executeSubtask
} from "./subtask/executeSubtask.js";


/*
 * =========================================================
 * RUN MULTIPLE SUBTASKS
 * =========================================================
 *
 * Реализация находится:
 *
 * core/subtask/runSubtasks.js
 *
 * Внутренняя цепочка:
 *
 * decomposition
 *   ↓
 * executeSubtask()
 *   ↓
 * results[]
 *   ↓
 * subtaskSummary
 *
 * =========================================================
 */


export {
    runSubtasks
} from "./subtask/runSubtasks.js";
