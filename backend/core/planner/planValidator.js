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
 * Проверяет нормализованный план Planner.
 *
 *
 * Контролирует:
 *
 * - структуру плана;
 * - intent;
 * - requiresTools;
 * - evidence;
 * - доступность инструментов;
 * - уникальность шагов;
 * - зависимости $from;
 * - соответствие PlanningContext.
 *
 *
 * НЕ отвечает за:
 *
 * - выполнение инструментов;
 * - проверку ответа AI;
 * - качество источника;
 * - обучение.
 *
 * =========================================================
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
        Array.isArray(plan)
    ) {

        return {

            success:
                false,

            text:
                "Некорректный план"

        };

    }


    if (
        typeof plan.intent !== "string" ||
        !plan.intent.trim()
    ) {

        return {

            success:
                false,

            text:
                "В плане отсутствует intent"

        };

    }


    if (
        typeof plan.requiresTools !== "boolean"
    ) {

        return {

            success:
                false,

            text:
                "В плане отсутствует requiresTools"

        };

    }


    if (
        !Array.isArray(plan.steps)
    ) {

        return {

            success:
                false,

            text:
                "В плане отсутствует steps"

        };

    }


    return {

        success:
            true

    };

}



/*
 * =========================================================
 * EXPERIENCE CONTEXT VALIDATION
 * =========================================================
 *
 * Проверяем согласованность
 * накопленного опыта Jessica
 * и выбранного маршрута Planner.
 *
 * Experience не заставляет Planner
 * действовать строго по Skill.
 *
 * Он только добавляет ограничения.
 *
 * =========================================================
 */


function validatePlanningContext(
    plan,
    context = {}
) {


    const experience =
        context?.experience;


    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return {

            success:
                true

        };

    }


    /*
     * Если Skill требует
     * подтверждения источников,
     * нельзя игнорировать evidence.
     */


    const validationRules =
        Array.isArray(
            experience.validationRules
        )
            ? experience.validationRules
            : [];


    const requiresSourceValidation =
        validationRules.some(
            rule =>
                String(rule)
                    .toLowerCase()
                    .includes("источник")
        );


    if (
        requiresSourceValidation &&
        plan.evidence?.mode === "none"
    ) {

        return {

            success:
                false,

            text:
                "Experience требует проверки источника, но Planner выбрал evidence.mode=none"

        };

    }


    return {

        success:
            true

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

            success:
                false,

            text:
                "При requiresTools=false steps должен быть пустым"

        };

    }


    if (
        plan.evidence?.mode !== "none"
    ) {

        return {

            success:
                false,

            text:
                "Задача без инструментов должна иметь evidence.mode=none"

        };

    }


    return {

        success:
            true

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

            success:
                false,

            text:
                "Planner не создал инструментальные шаги"

        };

    }


    if (
        plan.steps.length > MAX_STEPS
    ) {

        return {

            success:
                false,

            text:
                `Слишком много шагов: ${plan.steps.length}`

        };

    }



    const registeredTools =
        new Set(

            listTools()
                .map(
                    tool =>
                        tool.name
                )

        );



    const stepIds =
        new Set();



    /*
     * Проверяем структуру шагов.
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
            Array.isArray(step)
        ) {

            return {

                success:
                    false,

                text:
                    `Некорректный шаг ${index + 1}`

            };

        }



        if (
            typeof step.id !== "string" ||
            !step.id.trim()
        ) {

            return {

                success:
                    false,

                text:
                    `У шага ${index + 1} отсутствует id`

            };

        }



        if (
            stepIds.has(step.id)
        ) {

            return {

                success:
                    false,

                text:
                    `Повторяется id шага: ${step.id}`

            };

        }


        stepIds.add(
            step.id
        );



        if (
            typeof step.tool !== "string" ||
            !step.tool.trim()
        ) {

            return {

                success:
                    false,

                text:
                    `У шага ${step.id} отсутствует tool`

            };

        }



        if (
            !registeredTools.has(
                step.tool
            )
        ) {

            return {

                success:
                    false,

                text:
                    `Неизвестный инструмент: ${step.tool}`

            };

        }



        if (
            !step.arguments ||
            typeof step.arguments !== "object" ||
            Array.isArray(step.arguments)
        ) {

            return {

                success:
                    false,

                text:
                    `Некорректные arguments в шаге ${step.id}`

            };

        }


    }



    /*
     * Проверка ссылок между шагами.
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

                success:
                    false,

                text:
                    (
                        `Ошибка зависимостей шага ${step.id}: ` +
                        referenceValidation.text
                    )

            };

        }


        previousStepIds.add(
            step.id
        );

    }



    return {

        success:
            true

    };

}



/*
 * =========================================================
 * PUBLIC VALIDATION
 * =========================================================
 */


export function validatePlan(
    plan,
    context = {}
) {


    /*
     * 1. Структура
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
     * 2. Evidence
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
     * 3. Experience / PlanningContext
     */


    const contextValidation =
        validatePlanningContext(
            plan,
            context
        );


    if (
        !contextValidation.success
    ) {

        return contextValidation;

    }



    /*
     * 4. Без инструментов
     */


    if (
        plan.requiresTools === false
    ) {

        return validateNoToolsPlan(
            plan
        );

    }



    /*
     * 5. С инструментами
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
