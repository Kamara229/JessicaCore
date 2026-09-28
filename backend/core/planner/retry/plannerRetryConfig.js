/*
 * =========================================================
 * JESSICA PLANNER RETRY CONFIG
 * =========================================================
 *
 * Конфигурация retry-политики Planner.
 *
 * НЕ содержит бизнес-логику.
 * =========================================================
 */


export const MAX_PLANNER_ATTEMPTS =
    3;


export const INITIAL_RETRY_DELAY =
    1000;


export const MAX_RETRY_DELAY =
    10000;


export const RETRY_AFTER_BUFFER =
    250;


export const MAX_RETRY_AFTER =
    30000;
