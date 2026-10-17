/*
 * =========================================================
 * JESSICA ANSWER VALIDATION COORDINATOR v2
 * =========================================================
 *
 * Локальный semantic repair итогового ответа.
 *
 *
 * Flow:
 *
 * Answer
 *   ↓
 * Validation
 *   ↓
 * presentation failure?
 *   ├── no  → return validation result
 *   └── yes
 *        ↓
 *    Answer Repair
 *        ↓
 *    Validation
 *
 *
 * КРИТИЧЕСКО:
 *
 * presentation repair НЕ:
 *
 * - повторяет Task Runner;
 * - вызывает Tools;
 * - меняет Plan;
 * - запускает Replanner.
 *
 * =========================================================
 */


import {
    executeAnswer
} from "./answerExecutor.js";


import {
    executeValidation
} from "./validationExecutor.js";


const MAX_PRESENTATION_REPAIRS =
    1;


function isPresentationFailure(
    result
) {

    return (
        result?.success !== true
        &&
        result?.failure?.category === "presentation"
        &&
        result?.failure?.failureScope === "presentation"
    );

}


function buildRepairContext(
    context,
    validationFailure
) {

    return {

        previousAnswer:
            context?.answerResult?.text
            ||
            "",

        validatorReason:
            validationFailure?.reason
            ||
            "Итоговый ответ не соответствует данным выполнения."

    };

}


export async function executeAnswerValidation(

    context,

    executeStage

) {

    if(
        typeof executeStage !== "function"
    ){

        throw new Error(
            "Answer Validation Coordinator не получил executeStage"
        );

    }


    /*
     * =====================================================
     * INITIAL ANSWER
     * =====================================================
     */


    const answer =
        await executeStage(

            context,

            "composer",

            executeAnswer

        );


    if(
        answer?.success !== true
    ){

        return answer;

    }


    /*
     * =====================================================
     * INITIAL VALIDATION
     * =====================================================
     */


    let validation =
        await executeStage(

            context,

            "validator",

            executeValidation,

            {

                failureStatusResolver:
                    result =>
                        isPresentationFailure(
                            result
                        )
                            ? "REPAIR_REQUIRED"
                            : "FAILED"

            }

        );


    if(
        validation?.success === true
    ){

        return validation;

    }


    /*
     * =====================================================
     * PRESENTATION REPAIR
     * =====================================================
     */


    let repairCount =
        0;


    while(
        isPresentationFailure(
            validation
        )
        &&
        repairCount < MAX_PRESENTATION_REPAIRS
    ){

        repairCount += 1;


        context.metadata =
            context.metadata &&
            typeof context.metadata === "object"
                ? context.metadata
                : {};


        context.metadata.answerRepairCount =
            Number(
                context.metadata.answerRepairCount || 0
            )
            +
            1;


        const repairContext =
            buildRepairContext(

                context,

                validation.failure

            );


        console.log(

            "Jessica Answer Repair:",

            JSON.stringify({

                attempt:
                    repairCount,

                maxAttempts:
                    MAX_PRESENTATION_REPAIRS,

                reason:
                    repairContext.validatorReason

            })

        );


        const repairedAnswer =
            await executeStage(

                context,

                "composer-repair",

                currentContext =>
                    executeAnswer(

                        currentContext,

                        {
                            repairContext
                        }

                    )

            );


        if(
            repairedAnswer?.success !== true
        ){

            return repairedAnswer;

        }


        validation =
            await executeStage(

                context,

                "validator-repair",

                executeValidation

            );


        if(
            validation?.success === true
        ){

            console.log(

                "Jessica Answer Repair completed:",

                JSON.stringify({

                    attempt:
                        repairCount,

                    success:
                        true

                })

            );


            return validation;

        }

    }


    /*
     * Если presentation failure сохранился после
     * локального repair, он уходит в Failure Engine
     * как presentation и НЕ должен вызывать Replan.
     */


    return validation;

}
