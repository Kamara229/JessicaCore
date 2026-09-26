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
 * Проверка нормализованного Execution Plan.
 *
 *
 * Проверяет:
 *
 * - структуру;
 * - tools;
 * - evidence;
 * - зависимости;
 * - Experience compatibility.
 *
 *
 * НЕ:
 *
 * - выполняет план;
 * - меняет Experience;
 * - обучает Jessica.
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
                "В плане отсутствует intent"

        };

    }




    if (
        typeof plan.requiresTools !== "boolean"
    ) {


        return {

            success:false,

            text:
                "В плане отсутствует requiresTools"

        };

    }




    if (
        !Array.isArray(plan.steps)
    ) {


        return {

            success:false,

            text:
                "В плане отсутствует steps"

        };

    }




    return {

        success:true

    };


}









/*
 * =========================================================
 * EXPERIENCE VALIDATION
 * =========================================================
 */


function validateExperienceUsage(

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
     * Если Experience не использован,
     * ничего не проверяем.
     */


    if (
        plan.experienceUsed !== true
    ) {


        return {

            success:true

        };

    }







    const skills =


        Array.isArray(
            experience?.experience?.skills
        )

            ? experience.experience.skills

            : [];





    /*
     * План заявил использование опыта,
     * но Skills отсутствуют.
     */


    if (
        skills.length === 0
    ) {


        return {

            success:false,

            text:
                "Planner указал использование Experience, но Skill Context отсутствует"

        };

    }







    /*
     * Проверяем ограничения Skill.
     *
     * Сейчас только наличие.
     * Детальная проверка выполняется Execution/Learning.
     */


    for (
        const skill
        of skills
    ) {


        if (
            Array.isArray(
                skill.constraints
            )
            &&
            skill.constraints.length > 0
        ) {


            console.log(

                "Jessica Skill constraints:",

                {

                    skill:
                        skill.name ||
                        skill.id,


                    constraints:
                        skill.constraints

                }

            );

        }

    }




    return {

        success:true

    };


}









/*
 * =========================================================
 * LEGACY EXPERIENCE VALIDATION
 * =========================================================
 */


function validateLegacyExperience(

    plan,

    context = {}

) {


    const experience =
        context?.experience;



    if (
        !experience
    ) {

        return {

            success:true

        };

    }




    const rules =

        Array.isArray(
            experience.validationRules
        )

            ? experience.validationRules

            : [];





    const requiresSourceValidation =

        rules.some(

            rule =>

                String(rule)
                    .toLowerCase()
                    .includes(
                        "источник"
                    )

        );





    if (
        requiresSourceValidation &&

        plan.evidence?.mode === "none"

    ) {


        return {


            success:false,


            text:

            "Experience требует проверки источника, но evidence.mode=none"


        };


    }




    return {

        success:true

    };

}









/*
 * =========================================================
 * NO TOOLS
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
                "При requiresTools=false steps должен быть пустым"

        };

    }



    if (
        plan.evidence?.mode !== "none"
    ) {


        return {

            success:false,

            text:
                "Без инструментов evidence должен быть none"

        };

    }



    return {

        success:true

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


    if (
        plan.steps.length === 0
    ) {


        return {

            success:false,

            text:
                "Planner не создал шаги"

        };

    }




    if (
        plan.steps.length > MAX_STEPS
    ) {


        return {

            success:false,

            text:
                `Слишком много шагов: ${plan.steps.length}`

        };

    }





    const tools =

        new Set(

            listTools()

                .map(

                    tool =>
                        tool.name

                )

        );





    const ids =
        new Set();



    for (
        const step
        of plan.steps
    ) {



        if (
            !step.id
        ) {


            return {

                success:false,

                text:
                    "Шаг без id"

            };

        }




        if (
            ids.has(
                step.id
            )
        ) {


            return {

                success:false,

                text:
                    `Дублирующийся id: ${step.id}`

            };

        }



        ids.add(
            step.id
        );





        if (
            !tools.has(
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
            typeof step.arguments !== "object"
        ) {


            return {

                success:false,

                text:
                    `Некорректные arguments: ${step.id}`

            };

        }

    }






    const previousIds =
        new Set();





    for (
        const step
        of plan.steps
    ) {


        const validation =
            validateReferences(

                step.arguments,

                previousIds

            );



        if (
            !validation.success
        ) {


            return {

                success:false,

                text:
                    `Ошибка ссылки ${step.id}: ${validation.text}`

            };

        }



        previousIds.add(
            step.id
        );

    }



    return {

        success:true

    };

}









/*
 * =========================================================
 * PUBLIC
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







    const experience =
        validateExperienceUsage(

            plan,

            context

        );



    if (
        !experience.success
    ) {

        return experience;

    }






    const legacy =
        validateLegacyExperience(

            plan,

            context

        );



    if (
        !legacy.success
    ) {

        return legacy;

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









export function getMaxPlannerSteps() {

    return MAX_STEPS;

}
