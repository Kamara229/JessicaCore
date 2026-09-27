/*
 * =========================================================
 * JESSICA TASK RUNNER
 * STEP PREPARATION v1
 * =========================================================
 *
 * Подготовка Execution Step перед запуском.
 *
 *
 * Отвечает:
 *
 * - проверка структуры шага;
 * - получение stepId;
 * - проверка tool;
 * - проверка регистрации инструмента.
 *
 *
 * НЕ:
 *
 * - выполняет Tool;
 * - разрешает arguments;
 * - нормализует результат;
 * - принимает Retry/Replan.
 *
 * =========================================================
 */



import {
    getStepId
} from "./planRuntimeValidator.js";


import {
    hasTool
} from "../../tools/toolRegistry.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure({

    failureType,

    text,

    failedStep,

    failedStepId = null,

    results = []

}) {


    return {


        success:false,


        shouldRetry:false,


        needsClarification:false,


        stage:
            "runner",



        failureType,



        reason:
            text,



        text,



        failedStep,


        failedStepId,


        results


    };

}









/*
 * =========================================================
 * PREPARE STEP
 * =========================================================
 */


export function prepareStep(

    step,

    index,

    results = []

) {



    /*
     * =====================================================
     * STEP VALIDATION
     * =====================================================
     */


    if (

        !step ||

        typeof step !== "object" ||

        Array.isArray(step)

    ) {


        return buildFailure({

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









    if (

        !toolName

    ) {


        return buildFailure({

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

        !hasTool(

            toolName

        )

    ) {


        return buildFailure({

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









    return {


        success:true,


        stepId,


        toolName,


        step


    };

}
