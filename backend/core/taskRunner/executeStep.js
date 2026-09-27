/*
 * =========================================================
 * JESSICA TASK RUNNER
 * EXECUTE STEP v1
 * =========================================================
 *
 * Выполнение одного Execution Step.
 *
 *
 * Flow:
 *
 * Prepared Step
 *       ↓
 * Resolve Arguments
 *       ↓
 * Execute Tool
 *       ↓
 * Normalize Result
 *       ↓
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
    executeTool
} from "../../tools/toolRegistry.js";


import {
    resolveStepArguments
} from "./argumentResolver.js";


import {
    normalizeStepResult
} from "./stepResult.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure({

    stage,

    failureType,

    text,

    failedStep,

    failedStepId,

    results = []

}) {


    return {


        success:false,


        shouldRetry:false,


        needsClarification:false,


        stage,


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
 * EXECUTE STEP
 * =========================================================
 */


export async function executeStep({

    step,

    stepId,

    toolName,

    index,

    results,

    selectionContext

}) {



    /*
     * =====================================================
     * ORIGINAL ARGUMENTS
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









    /*
     * =====================================================
     * ARGUMENT RESOLUTION
     * =====================================================
     */


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
                `Ошибка подготовки аргументов шага ${stepId}`,


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


        console.error(

            `Jessica tool execution error [${toolName}]:`,

            error

        );



        return buildFailure({

            stage:
                "tool",


            failureType:
                "tool-exception",


            text:
                `Ошибка выполнения инструмента ${toolName}`,


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
     * CLARIFICATION
     * =====================================================
     */


    if (

        result.needsClarification === true

    ) {


        const reason =

            result.reason ||

            result.text ||

            "Требуется уточнение";



        return buildFailure({

            stage:
                result.stage ||
                "tool",


            failureType:
                result.failureType ||
                "needs-clarification",


            text:
                reason,


            failedStep:
                index,


            failedStepId:
                stepId,


            results

        });

    }









    /*
     * =====================================================
     * TOOL FAILURE
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
