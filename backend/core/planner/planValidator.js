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
 * JESSICA PLAN VALIDATOR v2
 * =========================================================
 *
 * Проверяет нормализованный Execution Plan.
 *
 *
 * Проверяет:
 *
 * - структуру плана;
 * - evidence;
 * - доступность tools;
 * - шаги;
 * - зависимости $from;
 * - совместимость PlanningContext.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - меняет план;
 * - вызывает AI;
 * - работает с Learning.
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

            success:false,

            text:
                "Некорректный план"

        };

    }





    if (
        typeof plan.intent !== "string" ||
        !plan.intent.trim()
    ) {


        return {

            success:false,

            text:
                "Отсутствует intent"

        };

    }





    if (
        typeof plan.requiresTools !== "boolean"
    ) {


        return {

            success:false,

            text:
                "Отсутствует requiresTools"

        };

    }





    if (
        !Array.isArray(plan.steps)
    ) {


        return {

            success:false,

            text:
                "Отсутствует steps"

        };

    }





    return {

        success:true

    };

}









/*
 * =========================================================
 * PLANNING CONTEXT COMPATIBILITY
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

            success:true

        };

    }





    /*
     * Если есть Experience,
     * проверяем только критические ограничения.
     *
     * Experience не управляет Planner.
     * Он только добавляет рекомендации.
     */



    const rules =

        Array.isArray(
            experience.validationRules
        )

            ? experience.validationRules

            : [];





    const sourceRequired =

        rules.some(

            rule =>

                String(rule)
                    .toLowerCase()
                    .includes(
                        "источник"
                    )

        );





    if (
        sourceRequired &&

        plan.evidence?.mode === "none"

    ) {


        return {


            success:false,


            text:

                "PlanningContext требует подтверждения источника"

        };


    }





    return {

        success:true

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

            success:false,

            text:
                "requiresTools=false не может иметь steps"

        };

    }





    if (
        plan.evidence?.mode !== "none"
    ) {


        return {

            success:false,

            text:
                "План без инструментов должен иметь evidence.mode=none"

        };

    }





    return {

        success:true

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

            success:false,

            text:
                "План требует tools, но steps отсутствуют"

        };

    }





    if (
        plan.steps.length > MAX_STEPS
    ) {


        return {

            success:false,

            text:
                `Превышен лимит шагов: ${plan.steps.length}`

        };

    }





    const availableTools =

        new Set(

            listTools()
                .map(

                    tool =>
                        tool.name

                )

        );





    const stepIds =
        new Set();



    const previousStepIds =
        new Set();







    for (
        const step
        of plan.steps
    ) {




        if (
            !step ||
            typeof step !== "object"
        ) {


            return {

                success:false,

                text:
                    "Некорректный step"

            };

        }








        if (
            !step.id ||
            typeof step.id !== "string"
        ) {


            return {

                success:false,

                text:
                    "Step без id"

            };

        }







        if (
            stepIds.has(
                step.id
            )
        ) {


            return {

                success:false,

                text:
                    `Повторяющийся id шага: ${step.id}`

            };

        }





        stepIds.add(
            step.id
        );







        if (
            !step.tool ||
            typeof step.tool !== "string"
        ) {


            return {

                success:false,

                text:
                    `У шага ${step.id} отсутствует tool`

            };

        }







        if (
            !availableTools.has(
                step.tool
            )
        ) {


            return {

                success:false,

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

                success:false,

                text:
                    `Некорректные arguments у ${step.id}`

            };

        }







        const referenceValidation =

            validateReferences(

                step.arguments,

                previousStepIds

            );





        if (
            !referenceValidation.success
        ) {


            return {

                success:false,

                text:

                    `Ошибка зависимости ${step.id}: ${referenceValidation.text}`

            };

        }






        previousStepIds.add(
            step.id
        );


    }





    return {

        success:true

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




    const basic =

        validateBasicStructure(
            plan
        );



    if (
        !basic.success
    ) {

        return basic;

    }








    const evidence =

        validateEvidencePlan(
            plan
        );



    if (
        !evidence.success
    ) {

        return evidence;

    }








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








    if (
        plan.requiresTools === false
    ) {


        return validateNoToolsPlan(
            plan
        );

    }








    return validateToolSteps(
        plan
    );


}









/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


export function getMaxPlannerSteps() {

    return MAX_STEPS;

}
