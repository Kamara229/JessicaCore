/*
 * =========================================================
 * JESSICA AI RETRY FACADE v3
 * =========================================================
 *
 * Публичный API технической retry-системы.
 *
 *
 * Внутренняя реализация:
 *
 * retry/
 *
 * ├── retryConstants.js
 * ├── retryError.js
 * ├── retryAfter.js
 * ├── retryClassifier.js
 * ├── retryDelay.js
 * └── retryExecutor.js
 *
 *
 * Внешние модули Jessica должны импортировать
 * retry API только через этот файл.
 *
 * =========================================================
 */


export {
    MAX_AI_ATTEMPTS
} from "./retry/retryConstants.js";


export {
    isRetryableAIError,
    isTransientModelProtocolError
} from "./retry/retryClassifier.js";


export {
    getAIRetryDelay,
    sleep
} from "./retry/retryDelay.js";


export {
    executeAIWithRetry
} from "./retry/retryExecutor.js";             
