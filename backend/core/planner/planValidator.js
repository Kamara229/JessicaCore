import {
    listTools
} from "../../tools/toolRegistry.js";

import {
    validateReferences
} from "./referenceValidator.js";

import {
    validateEvidencePlan
} from "./evidenceValidator.js";


/*
 * =========================================================
 * JESSICA PLAN VALIDATOR
 * =========================================================
 *
 * Проверяет уже нормализованный план.
 *
 * Здесь контролируется:
 *
 * - intent;
 * - requiresTools;
 * - steps;
 * - существование tools;
 * - уникальность id;
 * - ссылки $from;
 * - evidence.
 */


const MAX_STEPS =
    15;


/*
 * =========================================================
 * BASIC STRUCTURE
 * =========================================================
 */


function validateBasicStructure(
    plan
) {

    if (
        !plan ||
        typeof plan !== "object" ||
        Array.isArray(
            plan
        )
    ) {

        return {
            success: false,
            text:
                "Некорректный план"
        };

    }


    if (
        typeof plan.intent !== "string" ||
        !plan.intent.trim()
    ) {

        return {
            success: false,
            text:
                "В плане отсутствует intent"
        };

    }


    if (
        typeof plan.requiresTools !==
        "boolean"
    ) {

        return {
            success: false,
            text:
                "В плане отсутствует requiresTools"
        };

    }


    if (
        !Array.isArray(
            plan.steps
        )
    ) {

        return {
            success: false,
            text:
                "В плане отсутствует steps"
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * NO TOOLS PLAN
 * =========================================================
 */


function validateNoToolsPlan(
    plan
) {

    if (
        plan.steps.length !== 0
    ) {

        return {
            success: false,
            text:
                "При requiresTools=false steps должен быть пустым"
        };

    }


    if (
        plan.evidence?.mode !==
        "none"
    ) {

        return {
            success: false,
            text:
                "Задача без инструментов должна иметь evidence.mode=none"
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * TOOL STEPS
 * =========================================================
 */


function validateToolSteps(
    plan
) {

    if (
        plan.steps.length === 0
    ) {

        return {
            success: false,
            text:
                "Planner не создал инструментальные шаги"
        };

    }


    if (
        plan.steps.length >
        MAX_STEPS
    ) {

        return {
            success: false,
            text:
                `Слишком много шагов: ${plan.steps.length}`
        };

    }


    const registeredToolNames =
        new Set(
            listTools().map(
                tool =>
                    tool.name
            )
        );


    const allIds =
        new Set();


    /*
     * Сначала проверяем каждый шаг
     * и уникальность id.
     */
    for (
        let index = 0;
        index < plan.steps.length;
        index++
    ) {

        const step =
            plan.steps[index];


        if (
            !step ||
            typeof step !== "object" ||
            Array.isArray(
                step
            )
        ) {

            return {
                success: false,
                text:
                    `Некорректный шаг ${index + 1}`
            };

        }


        if (
            typeof step.id !== "string" ||
            !step.id.trim()
        ) {

            return {
                success: false,
                text:
                    `У шага ${index + 1} отсутствует id`
            };

        }


        if (
            allIds.has(
                step.id
            )
        ) {

            return {
                success: false,
                text:
                    `Повторяется id шага: ${step.id}`
            };

        }


        allIds.add(
            step.id
        );


        if (
            typeof step.tool !== "string" ||
            !step.tool.trim()
        ) {

            return {
                success: false,
                text:
                    `У шага ${step.id} отсутствует tool`
            };

        }


        if (
            !registeredToolNames.has(
                step.tool
            )
        ) {

            return {
                success: false,
                text:
                    `Неизвестный инструмент: ${step.tool}`
            };

        }


        if (
            !step.arguments ||
            typeof step.arguments !== "object" ||
            Array.isArray(
                step.arguments
            )
        ) {

            return {
                success: false,
                text:
                    `Некорректные arguments в шаге ${step.id}`
            };

        }

    }


    /*
     * Теперь проверяем зависимости.
     *
     * $from может вести только
     * на уже предыдущий шаг.
     */
    const previousStepIds =
        new Set();


    for (
        const step
        of plan.steps
    ) {

        const referenceValidation =
            validateReferences(
                step.arguments,
                previousStepIds
            );


        if (
            !referenceValidation.success
        ) {

            return {
                success: false,
                text:
                    (
                        `Ошибка зависимостей шага ${step.id}: ` +
                        `${referenceValidation.text}`
                    )
            };

        }


        previousStepIds.add(
            step.id
        );

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * PUBLIC VALIDATION
 * =========================================================
 */


export function validatePlan(
    plan
) {

    /*
     * 1. Общая структура.
     */
    const basicValidation =
        validateBasicStructure(
            plan
        );


    if (
        !basicValidation.success
    ) {

        return basicValidation;

    }


    /*
     * 2. Evidence.
     */
    const evidenceValidation =
        validateEvidencePlan(
            plan
        );


    if (
        !evidenceValidation.success
    ) {

        return evidenceValidation;

    }


    /*
     * 3. План без tools.
     */
    if (
        plan.requiresTools === false
    ) {

        return validateNoToolsPlan(
            plan
        );

    }


    /*
     * 4. Инструментальный план.
     */
    return validateToolSteps(
        plan
    );

}


/*
 * =========================================================
 * CONFIG INFO
 * =========================================================
 */


export function getMaxPlannerSteps() {

    return MAX_STEPS;

}
