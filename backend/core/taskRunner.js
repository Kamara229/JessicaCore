import {
    executeTool,
    hasTool
} from "../tools/toolRegistry.js";

import {
    validatePlanForExecution,
    getStepId
} from "./taskRunner/planRuntimeValidator.js";

import {
    resolveStepArguments
} from "./taskRunner/argumentResolver.js";

import {
    normalizeStepResult
} from "./taskRunner/stepResult.js";


/*
 * =========================================================
 * JESSICA TASK RUNNER
 * =========================================================
 *
 * Центральный координатор выполнения плана.
 *
 *
 * Flow:
 *
 * Plan
 *  ↓
 * Runtime Validation
 *  ↓
 * Resolve Arguments
 *  ↓
 * Execute Tool
 *  ↓
 * Normalize Result
 *  ↓
 * Next Step
 *
 *
 * TaskRunner НЕ:
 *
 * - строит планы;
 * - вызывает Planner;
 * - вызывает Replanner;
 * - принимает решение о новом маршруте;
 * - валидирует финальный ответ;
 * - работает с Experience.
 *
 *
 * Если текущий маршрут можно перестроить,
 * TaskRunner сообщает:
 *
 * shouldRetry: true
 *
 * Execution Cycle принимает дальнейшее решение.
 *
 * =========================================================
 */


/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function buildFailure({

    stage =
        "runner",

    failureType =
        "run-failure",

    text =
        "Не удалось выполнить план",

    reason =
        "",

    shouldRetry =
        false,

    needsClarification =
        false,

    failedStep =
        null,

    failedStepId =
        null,

    results =
        []

}) {


    const finalReason =
        reason ||
        text ||
        "Не удалось выполнить план";


    return {

        success:
            false,


        shouldRetry:
            shouldRetry === true,


        needsClarification:
            needsClarification === true,


        stage,


        failureType,


        reason:
            finalReason,


        text:
            text ||
            finalReason,


        failedStep,


        failedStepId,


        results

    };

}



/*
 * =========================================================
 * SUCCESS RESULT
 * =========================================================
 */


function buildSuccess(
    results
) {

    return {

        success:
            true,

        shouldRetry:
            false,

        needsClarification:
            false,

        text:
            "План выполнен",

        results

    };

}



/*
 * =========================================================
 * RUN PLAN
 * =========================================================
 */


export async function runPlan(

    plan,

    task = ""

) {


    /*
     * =====================================================
     * 1. RUNTIME PLAN VALIDATION
     * =====================================================
     */


    const planValidation =
        validatePlanForExecution(
            plan
        );



    if (
        !planValidation.success
    ) {

        return buildFailure({

            stage:
                planValidation.stage ||
                "runner",

            failureType:
                planValidation.failureType ||
                "invalid-plan",

            text:
                planValidation.text ||
                "Некорректный план",

            shouldRetry:
                planValidation.shouldRetry === true,

            results:
                []

        });

    }



    /*
     * =====================================================
     * NO TOOLS
     * =====================================================
     */


    if (
        planValidation.noTools === true
    ) {

        return {

            success:
                true,

            shouldRetry:
                false,

            needsClarification:
                false,

            text:
                "Инструменты не требуются",

            results:
                []

        };

    }



    /*
     * =====================================================
     * 2. SOURCE SELECTION CONTEXT
     * =====================================================
     */


    const selectionContext =

        String(
            task || ""
        )
        .trim()

        ||

        [

            plan?.intent || "",

            plan?.reasoningSummary || "",

            plan?.evidence?.reason || ""

        ]
        .filter(Boolean)
        .join("\n");



    /*
     * =====================================================
     * 3. EXECUTION
     * =====================================================
     */


    const results =
        [];



    for (
        let index = 0;

        index < plan.steps.length;

        index++
    ) {


        /*
         * =================================================
         * STEP
         * =================================================
         */


        const originalStep =
            plan.steps[index];



        if (
            !originalStep ||
            typeof originalStep !== "object" ||
            Array.isArray(originalStep)
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

                originalStep,

                index

            );



        /*
         * =================================================
         * TOOL
         * =================================================
         */


        const toolName =
            typeof originalStep.tool === "string"

                ? originalStep.tool.trim()

                : "";



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
            !hasTool(
                toolName
            )
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
         * =================================================
         * ORIGINAL ARGUMENTS
         * =================================================
         */


        const originalArgs =

            originalStep.arguments &&

            typeof originalStep.arguments === "object" &&

            !Array.isArray(
                originalStep.arguments
            )

                ? originalStep.arguments

                : {};



        /*
         * =================================================
         * 4. RESOLVE ARGUMENTS
         * =================================================
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


        } catch (error) {


            console.error(

                `Jessica TaskRunner argument exception [${stepId}]:`,

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



        /*
         * =================================================
         * ARGUMENT / SOURCE FAILURE
         * =================================================
         */


        if (
            !resolvedArgsResult?.success
        ) {


            const failureReason =

                resolvedArgsResult?.reason ||

                resolvedArgsResult?.text ||

                `Не удалось подготовить аргументы шага ${stepId}`;



            console.warn(

                "Jessica TaskRunner route failure:",

                JSON.stringify({

                    stage:
                        resolvedArgsResult?.stage ||
                        "argument-resolution",

                    failureType:
                        resolvedArgsResult?.failureType ||
                        "argument-resolution",

                    shouldRetry:
                        resolvedArgsResult?.shouldRetry === true,

                    reason:
                        failureReason

                })

            );



            return buildFailure({

                stage:
                    resolvedArgsResult?.stage ||
                    "argument-resolution",

                failureType:
                    resolvedArgsResult?.failureType ||
                    "argument-resolution",

                text:
                    failureReason,

                reason:
                    failureReason,

                shouldRetry:
                    resolvedArgsResult?.shouldRetry === true,

                needsClarification:
                    resolvedArgsResult?.needsClarification === true,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results

            });

        }



        const resolvedArgs =
            resolvedArgsResult.value;



        console.log(

            (
                `Jessica TaskRunner: ` +
                `step ${index + 1}/${plan.steps.length} ` +
                `[${stepId}] -> ${toolName}`
            )

        );



        /*
         * =================================================
         * 5. EXECUTE TOOL
         * =================================================
         */


        let rawResult;



        try {


            rawResult =
                await executeTool(

                    toolName,

                    resolvedArgs

                );


        } catch (error) {


            console.error(

                `Jessica TaskRunner tool exception [${toolName}]:`,

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
         * =================================================
         * 6. NORMALIZE TOOL RESULT
         * =================================================
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
         * =================================================
         * 7. NEEDS CLARIFICATION
         * =================================================
         */


        if (
            result.needsClarification === true
        ) {


            const clarificationReason =

                result.reason ||

                result.text ||

                "Для выполнения задачи требуется уточнение.";



            return buildFailure({

                stage:
                    result.stage ||
                    "tool",

                failureType:
                    result.failureType ||
                    "needs-clarification",

                text:
                    clarificationReason,

                reason:
                    clarificationReason,

                shouldRetry:
                    false,

                needsClarification:
                    true,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results

            });

        }



        /*
         * =================================================
         * 8. TOOL FAILURE
         * =================================================
         */


        if (
            result.success !== true
        ) {


            const failureReason =

                result.reason ||

                result.text ||

                `Не удалось выполнить шаг ${index + 1}`;



            console.warn(

                "Jessica TaskRunner tool failure:",

                JSON.stringify({

                    step:
                        stepId,

                    tool:
                        toolName,

                    shouldRetry:
                        result.shouldRetry === true,

                    stage:
                        result.stage ||
                        "tool",

                    failureType:
                        result.failureType ||
                        "tool-failure",

                    reason:
                        failureReason

                })

            );



            return buildFailure({

                stage:
                    result.stage ||
                    "tool",

                failureType:
                    result.failureType ||
                    "tool-failure",

                text:
                    failureReason,

                reason:
                    failureReason,

                shouldRetry:
                    result.shouldRetry === true,

                failedStep:
                    index,

                failedStepId:
                    stepId,

                results

            });

        }

    }



    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return buildSuccess(
        results
    );

}
