/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST CONSTANTS
 * =========================================================
 *
 * E2E PHASE 2
 *
 * Второе независимое наблюдение
 * той же Candidate lineage.
 *
 * Первый запуск:
 *
 * Blender
 * trace-v1
 *      ↓
 * KEEP_CANDIDATE
 *
 *
 * Второй запуск:
 *
 * Python
 * trace-v2
 *      ↓
 * merge existing Candidate
 *      ↓
 * occurrences = 2
 * confidence ≈ 0.82
 *      ↓
 * AUTO_APPROVE
 *
 * =========================================================
 */


export const AUTONOMOUS_TEST_SKILL_ID =
    "jessica-diagnostic-autonomous-learning";


export const AUTONOMOUS_TEST_SKILL_NAME =
    "Jessica Diagnostic Autonomous Learning";


/*
 * Новый Trace ID принципиален.
 *
 * Candidate Memory не должна считать
 * второй execution повторной обработкой
 * первого Trace.
 */


export const AUTONOMOUS_TEST_TRACE_ID =
    "diagnostic-autonomous-learning-trace-v2";


export const AUTONOMOUS_TEST_FLAG =
    "RUN_AUTONOMOUS_LEARNING_E2E_TEST_ON_START";


/*
 * Семантически тот же reusable pattern:
 *
 * найти и проверить официальный источник.
 *
 * Но задача другая, поэтому evidence
 * действительно независимый.
 */


export const AUTONOMOUS_TEST_TASK =
    "Найди официальный сайт Python и укажи проверенную ссылку";


export const AUTONOMOUS_TEST_RESULT =
    "Официальный сайт Python: https://www.python.org/";
