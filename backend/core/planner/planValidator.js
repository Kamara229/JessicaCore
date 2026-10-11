import {
    listTools
} from "../../tools/toolRegistry.js";


import {
    validateReferences
} from "./referenceValidator.js";


import {
    validateEvidencePlan
} from "./evidenceValidator.js";


import {
    validatePlannerEvidencePolicy
} from "./plannerEvidencePolicy.js";


/*
 * =========================================================
 * JESSICA PLAN VALIDATOR v4
 * =========================================================
 *
 * Финальная проверка Execution Plan.
 *
 *
 * Проверяет:
 *
 * - структуру;
 * - intent;
 * - requiresTools;
 * - evidence structure;
 * - evidence strength for user task;
 * - доступные инструменты;
 * - уникальность шагов;
 * - зависимости $from.
 *
 * =========================================================
 */


const MAX_STEPS =
    15;


/*
 * =========================================================
 * BASIC VALIDATION
 * =========================================================
 */


function validateBasicStructure(
    plan
) {

    if(
        !plan
        ||
        typeof plan !== "object"
        ||
        Array.isArray(plan)
    ){

        return {

            success:
                false,

            text:
                "Plan должен быть объектом"

        };

    }


    if(
        typeof plan.intent !== "string"
        ||
        !plan.intent.trim()
    ){

        return {

            success:
                false,

            text:
                "Не указан intent"

        };

    }


    if(
        typeof plan.requiresTools !== "boolean"
    ){

        return {

            success:
                false,

            text:
                "Не указан requiresTools"

        };

    }


    if(
        !Array.isArray(
            plan.steps
        )
    ){

        return {

            success:
                false,

            text:
                "steps должен быть массивом"

        };

    }


    return {

        success:
            true

    };

}


/*
 * =========================================================
 * NO TOOL PLAN
 * =========================================================
 */


function validateNoToolsPlan(
    plan
) {

    if(
        plan.steps.length > 0
    ){

        return {

            success:
                false,

            text:
                "requiresTools=false не допускает steps"

        };

    }


    if(
        plan.evidence?.mode !== "none"
    ){

        return {

            success:
                false,

            text:
                "Без инструментов evidence должен быть none"

        };

    }


    return {

        success:
            true

    };

}


/*
 * =========================================================
 * TOOL VALIDATION
 * =========================================================
 */


function validateToolSteps(
    plan
) {

    if(
        plan.steps.length === 0
    ){

        return {

            success:
                false,

            text:
                "requiresTools=true, но steps пустой"

        };

    }


    if(
        plan.steps.length > MAX_STEPS
    ){

        return {

            success:
                false,

            text:
                `Количество шагов превышает лимит ${MAX_STEPS}`

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


    const ids =
        new Set();


    const previousIds =
        new Set();


    for(
        const step
        of plan.steps
    ){

        if(
            !step
            ||
            typeof step !== "object"
        ){

            return {

                success:
                    false,

                text:
                    "Некорректный step"

            };

        }


        if(
            typeof step.id !== "string"
            ||
            !step.id.trim()
        ){

            return {

                success:
                    false,

                text:
                    "Step должен иметь id"

            };

        }


        if(
            ids.has(
                step.id
            )
        ){

            return {

                success:
                    false,

                text:
                    `Дублирующийся step id: ${step.id}`

            };

        }


        ids.add(
            step.id
        );


        if(
            typeof step.tool !== "string"
            ||
            !step.tool.trim()
        ){

            return {

                success:
                    false,

                text:
                    `У шага ${step.id} отсутствует tool`

            };

        }


        if(
            !registeredTools.has(
                step.tool
            )
        ){

            return {

                success:
                    false,

                text:
                    `Инструмент не зарегистрирован: ${step.tool}`

            };

        }


        if(
            !step.arguments
            ||
            typeof step.arguments !== "object"
            ||
            Array.isArray(
                step.arguments
            )
        ){

            return {

                success:
                    false,

                text:
                    `Некорректные arguments у ${step.id}`

            };

        }


        const referenceResult =

            validateReferences(

                step.arguments,

                previousIds

            );


        if(
            !referenceResult.success
        ){

            return {

                success:
                    false,

                text:

                    `Ошибка зависимости ${step.id}: ${referenceResult.text}`

            };

        }


        previousIds.add(
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
 * PUBLIC VALIDATOR
 * =========================================================
 */


export function validatePlan(

    plan,

    context = {},

    task = ""

) {

    /*
     * =====================================================
     * 1. STRUCTURE
     * =====================================================
     */


    const structure =

        validateBasicStructure(
            plan
        );


    if(
        !structure.success
    ){

        return structure;

    }


    /*
     * =====================================================
     * 2. EVIDENCE INTERNAL CONSISTENCY
     * =====================================================
     */


    const evidence =

        validateEvidencePlan(
            plan
        );


    if(
        !evidence.success
    ){

        return evidence;

    }


    /*
     * =====================================================
     * 3. EVIDENCE STRENGTH POLICY
     * =====================================================
     *
     * Здесь проверяем уже не сам Plan,
     * а достаточность evidence
     * для исходной пользовательской задачи.
     *
     * =====================================================
     */


    const evidencePolicy =

        validatePlannerEvidencePolicy({

            task:

                task

                ||

                context?.task

                ||

                "",

            plan

        });


    if(
        !evidencePolicy.success
    ){

        return evidencePolicy;

    }


    /*
     * =====================================================
     * 4. NO TOOLS
     * =====================================================
     */


    if(
        plan.requiresTools === false
    ){

        return validateNoToolsPlan(
            plan
        );

    }


    /*
     * =====================================================
     * 5. TOOL PLAN
     * =====================================================
     */


    return validateToolSteps(
        plan
    );

}


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


export function getMaxPlannerSteps()
{

    return MAX_STEPS;

}
