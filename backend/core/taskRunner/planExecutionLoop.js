/*
 * =========================================================
 * JESSICA TASK RUNNER
 * PLAN EXECUTION LOOP v1
 * =========================================================
 *
 * Выполнение списка шагов плана.
 *
 *
 * Flow:
 *
 * Plan Steps
 *      ↓
 * Step Validation
 *      ↓
 * Execute Step
 *      ↓
 * Collect Results
 *
 *
 * НЕ:
 *
 * - строит план;
 * - выполняет Retry;
 * - делает Replan;
 * - валидирует Answer.
 *
 * =========================================================
 */


import {
    hasTool
} from "../../tools/toolRegistry.js";


import {
    getStepId
} from "./planRuntimeValidator.js";


import {
    executeStep
} from "./executeStep.js";


import {
    buildFailure
} from "./resultHandler.js";









/*
 * =========================================================
 * RUN LOOP
 * =========================================================
 */


export async function executePlanSteps({

    plan,

    task,

    results = []

}) {


    const selectionContext =

        String(
            task || ""
        )
        .trim()

        ||

        [

            plan?.intent,

            plan?.reasoningSummary,

            plan?.evidence?.reason

        ]

        .filter(Boolean)

        .join("\n");









    for (

        let index = 0;

        index < plan.steps.length;

        index++

    ) {



        const step =

            plan.steps[index];









        /*
         * =====================================================
         * STEP CHECK
         * =====================================================
         */


        if (

            !step

            ||

            typeof step !== "object"

            ||

            Array.isArray(step)

        ) {


            return buildFailure({

                stage:
                    "runner",


                failureType:
                    "invalid-step",


                text:

                    `Некорректный шаг ${index + 1}`,


                failedStep:
                    index,


                results


            });

        }









        const stepId =

            getStepId(

                step,

                index

            );









        /*
         * =====================================================
         * TOOL CHECK
         * =====================================================
         */


        const toolName =

            typeof step.tool === "string"

                ?

                step.tool.trim()

                :

                "";









        if (!toolName) {


            return buildFailure({

                stage:
                    "runner",


                failureType:
                    "missing-tool",


                text:

                    `В шаге ${index + 1} отсутствует tool`,


                failedStep:
                    index,


                failedStepId:
                    stepId,


                results


            });


        }









        if (

            !hasTool(toolName)

        ) {


            return buildFailure({

                stage:
                    "runner",


                failureType:
                    "unknown-tool",


                text:

                    `Инструмент ${toolName} не зарегистрирован`,


                failedStep:
                    index,


                failedStepId:
                    stepId,


                results


            });


        }









        /*
         * =====================================================
         * EXECUTE STEP
         * =====================================================
         */


        const result =

            await executeStep({

                step,

                stepId,

                toolName,

                index,

                results,

                selectionContext


            });









        /*
         * Ошибка шага
         */


        if (

            result?.success !== true

        ) {


            return result;


        }









        /*
         * executeStep уже добавляет
         * результат в массив.
         *
         * Но если в будущем это изменится,
         * здесь будет единая точка контроля.
         */


    }









    return {


        success:true,


        results


    };


}
