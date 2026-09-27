/*
 * =========================================================
 * JESSICA TASK RUNNER
 * EXECUTE STEP v2
 * =========================================================
 *
 * Выполнение одного Execution Step.
 *
 *
 * Flow:
 *
 * Step
 *   ↓
 * Step Identity
 *   ↓
 * Tool Check
 *   ↓
 * Resolve Arguments
 *   ↓
 * Execute Tool
 *   ↓
 * Normalize Result
 *   ↓
 * Step Result
 *
 *
 * НЕ:
 *
 * - управляет циклом;
 * - делает Retry;
 * - делает Replan;
 * - меняет Execution Context.
 *
 * =========================================================
 */


import {
    executeTool,
    hasTool
} from "../../tools/toolRegistry.js";


import {
    getStepId
} from "./planRuntimeValidator.js";


import {
    resolveStepArguments
} from "./argumentResolver.js";


import {
    normalizeStepResult
} from "./stepResult.js";


import {
    buildFailure
} from "./resultHandler.js";









/*
 * =========================================================
 * EXECUTE STEP
 * =========================================================
 */


export async function executeStep({

    step,

    index,

    results,

    selectionContext

}) {



    /*
     * =====================================================
     * STEP IDENTITY
     * =====================================================
     */


    const stepId =

        getStepId(

            step,

            index

        );





    const toolName =

        typeof step?.tool === "string"

            ?

            step.tool.trim()

            :

            "";









    /*
     * =====================================================
     * STEP VALIDATION
     * =====================================================
     */


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
     * ARGUMENTS
     * =====================================================
     */


    const originalArgs =

        step.arguments &&

        typeof step.arguments === "object" &&

        !Array.isArray(step.arguments)

            ?

            step.arguments

            :

            {};







    let resolvedArgsResult;



    try {


        resolvedArgsResult =

            await resolveStepArguments(

                toolName,

                originalArgs,

                results,

                selectionContext

            );


    }

    catch(error){


        console.error(

            `Jessica argument resolver error [${stepId}]:`,

            error

        );



        return buildFailure({

            stage:
                "argument-resolution",


            failureType:
                "argument-resolution-error",


            text:

                error?.message ||

                `Ошибка подготовки аргументов ${stepId}`,


            failedStep:
                index,


            failedStepId:
                stepId,


            results

        });


    }









    if (

        resolvedArgsResult?.success !== true

    ) {


        return {


            ...resolvedArgsResult,


            failedStep:
                index,


            failedStepId:
                stepId,


            results

        };


    }









    const resolvedArgs =

        resolvedArgsResult.value;









    console.log(

        `Jessica TaskRunner: step ${index + 1} [${stepId}] -> ${toolName}`

    );









    /*
     * =====================================================
     * TOOL EXECUTION
     * =====================================================
     */


    let rawResult;



    try {


        rawResult =

            await executeTool(

                toolName,

                resolvedArgs

            );


    }

    catch(error){


        return buildFailure({

            stage:
                "tool",


            failureType:
                "tool-exception",


            text:

                error?.message ||

                `Ошибка выполнения ${toolName}`,


            failedStep:
                index,


            failedStepId:
                stepId,


            results

        });


    }









    /*
     * =====================================================
     * NORMALIZE
     * =====================================================
     */


    const result =

        normalizeStepResult(

            {

                id:
                    stepId,


                tool:
                    toolName,


                arguments:
                    resolvedArgs


            },


            rawResult

        );









    results.push(
        result
    );









    /*
     * =====================================================
     * STEP FAILURE
     * =====================================================
     */


    if (

        result.success !== true

    ) {


        return buildFailure({

            stage:

                result.stage ||

                "tool",



            failureType:

                result.failureType ||

                "tool-failure",



            text:

                result.reason ||

                result.text ||

                `Ошибка шага ${index + 1}`,



            failedStep:

                index,



            failedStepId:

                stepId,



            results


        });


    }









    return result;

}
