/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v4
 * =========================================================
 *
 * Координатор перестроения Execution Plan.
 *
 *
 * Flow:
 *
 * Failure Decision
 *        ↓
 * Replan Feedback
 *        ↓
 * Replanner
 *        ↓
 * Alternative Plan
 *        ↓
 * Apply Context
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - решает нужен ли Replan;
 * - выполняет инструменты;
 * - запускает Execution Cycle;
 * - меняет Experience.
 *
 * =========================================================
 */


import {
    replanTask
} from "../replanner.js";









/*
 * =========================================================
 * NORMALIZE STRING
 * =========================================================
 */


function normalizeString(
    value
) {

    return String(
        value || ""
    )
        .trim();

}









/*
 * =========================================================
 * BUILD VALIDATOR FEEDBACK
 * =========================================================
 *
 * Используется, если источник ошибки Validator.
 *
 * =========================================================
 */


export function buildValidatorFeedback(
    validation
) {


    return {


        stage:
            "validator",



        failureType:
            "validation-error",



        reason:

            normalizeString(
                validation?.reason
            )

            ||

            "Результат не прошёл проверку"


    };

}









/*
 * =========================================================
 * NORMALIZE REPLAN INPUT
 * =========================================================
 */


function normalizeReplanInput(
    feedback
) {


    if (
        !feedback ||
        typeof feedback !== "object"
    ) {


        return {


            stage:
                "execution",


            failureType:
                "unknown",


            reason:
                "Неизвестная причина перестроения"

        };

    }





    return {


        stage:

            normalizeString(
                feedback.stage
            )

            ||

            "execution",




        failureType:

            normalizeString(
                feedback.failureType
            )

            ||

            "execution-error",




        reason:

            normalizeString(
                feedback.reason
            )

            ||

            "Текущий план оказался неэффективным"


    };

}









/*
 * =========================================================
 * BUILD REPLAN CONTEXT
 * =========================================================
 */


function buildReplanContext(

    context,

    feedback

) {


    return {


        previousPlan:

            context?.plan || null,



        executionFailure:

            feedback,



        attempt:

            Number(
                context?.attempt || 0
            ),



        executionResults:

            context?.runResult || null


    };

}









/*
 * =========================================================
 * CREATE ALTERNATIVE PLAN
 * =========================================================
 */


export async function createAlternativePlan(

    context,

    feedback

) {


    const normalizedFeedback =

        normalizeReplanInput(
            feedback
        );





    console.log(

        "Jessica Replan:",

        {

            stage:
                normalizedFeedback.stage,


            failureType:
                normalizedFeedback.failureType,


            reason:
                normalizedFeedback.reason,


            attempt:
                context?.attempt || 0

        }

    );








    try {


        const result =

            await replanTask(

                context?.task || "",



                context?.plan || null,



                normalizedFeedback,



                context?.runResult || null,



                {


                    ...(context?.planningContext || {}),



                    replanContext:

                        buildReplanContext(

                            context,

                            normalizedFeedback

                        )

                }

            );









        if (
            !result?.success ||
            !result?.plan
        ) {


            return {


                success:false,


                reason:

                    result?.reason

                    ||

                    "Replanner не создал новый план"


            };

        }








        return {


            success:true,



            plan:

                result.plan,



            planningContext:

                result.context

                &&

                typeof result.context === "object"

                    ? result.context

                    :

                    context?.planningContext || {}



        };





    } catch(error) {



        console.error(

            "Jessica Replan error:",

            error

        );



        return {


            success:false,


            reason:

                error?.message

                ||

                "Ошибка Replanner"


        };


    }


}









/*
 * =========================================================
 * APPLY ALTERNATIVE PLAN
 * =========================================================
 *
 * Меняет только маршрут.
 *
 * Старые результаты удаляются,
 * потому что относятся к предыдущему плану.
 *
 * =========================================================
 */


export function applyAlternativePlan(

    context,

    alternative

) {


    if (

        !context ||

        !alternative?.success ||

        !alternative.plan

    ) {


        return false;

    }







    context.plan =

        alternative.plan;







    if (
        alternative.planningContext
    ) {


        context.planningContext =

            alternative.planningContext;


    }







    context.runResult =
        null;



    context.answerResult =
        null;



    context.validationResult =
        null;







    context.replanCount =

        Number(
            context.replanCount || 0
        )

        +

        1;







    return true;

}
