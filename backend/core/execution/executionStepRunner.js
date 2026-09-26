/*
 * =========================================================
 * JESSICA EXECUTION STEP RUNNER
 * =========================================================
 *
 * Один полный цикл выполнения:
 *
 * Plan
 *   ↓
 * Runner
 *   ↓
 * Composer
 *   ↓
 * Validator
 *
 *
 * Передаёт:
 *
 * ExecutionContext
 *        ↓
 * ExecutionResult
 *        ↓
 * Learning
 *
 *
 * НЕ отвечает за:
 *
 * - retry;
 * - replan;
 * - лимиты;
 * - terminal decision.
 *
 * =========================================================
 */



import {
    runPlan
} from "../taskRunner.js";


import {
    composeAnswer
} from "../answerComposer.js";


import {
    validateResult
} from "../validator.js";


import {
    analyzeRunFailure
} from "./runFailurePolicy.js";


import {
    buildCompletedResult,
    buildFailureResult
} from "./executionResult.js";









/*
 * =========================================================
 * EXPERIENCE META
 * =========================================================
 */


function buildExecutionExperienceMeta(
    context
) {


    return {


        used:

            context?.experience?.used === true,



        skills:

            context?.experience?.skills || [],



        skillIds:

            Array.isArray(
                context?.experience?.skills
            )

                ? context.experience.skills

                    .map(

                        skill =>

                            skill?.id ||
                            skill

                    )

                    .filter(Boolean)

                : []

    };

}









/*
 * =========================================================
 * EXECUTE STEP
 * =========================================================
 */


export async function executeExecutionStep(
    context
) {



    /*
     * =====================================================
     * RUN PLAN
     * =====================================================
     */


    try {


        context.runResult =

            await runPlan(

                context.plan,

                context.task

            );


    } catch(error) {


        console.error(
            "Jessica TaskRunner error:",
            error
        );


        return {


            success:false,


            failure:{

                stage:
                    "runner",


                reason:

                    error?.message ||

                    "Ошибка выполнения плана",



                failureType:

                    "runner-exception"

            }


        };


    }








    /*
     * =====================================================
     * ANALYZE RUN
     * =====================================================
     */


    const runFailure =

        analyzeRunFailure(

            context.runResult

        );




    if (
        runFailure?.failed === true
    ) {


        return {


            success:false,


            failure:
                runFailure


        };


    }








    /*
     * =====================================================
     * COMPOSE
     * =====================================================
     */


    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );



    } catch(error) {


        return {


            success:false,


            failure:{

                stage:
                    "composer",


                reason:

                    error?.message ||

                    "Ошибка создания ответа",



                failureType:

                    "composer-exception"

            }


        };


    }






    if (
        !context.answerResult?.success
    ) {


        return {


            success:false,


            failure:{

                stage:
                    "composer",


                reason:

                    context.answerResult?.text ||

                    "Ответ не создан",



                failureType:

                    "composer-failure"

            }


        };


    }








    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    let validation;



    try {


        validation =

            await validateResult(

                context.task,

                context.plan,

                context.runResult,

                context.answerResult

            );



    } catch(error) {



        return {


            success:true,


            result:

                buildCompletedResult(

                    context,

                    context.answerResult,

                    false

                )


        };


    }









    /*
     * =====================================================
     * VALID RESULT
     * =====================================================
     */


    if (
        validation?.valid === true
    ) {



        const result =

            buildCompletedResult(

                context,

                context.answerResult,

                true

            );





        result.executionMeta = {


            ...(result.executionMeta || {}),



            experience:

                buildExecutionExperienceMeta(
                    context
                )

        };





        return {


            success:true,


            result


        };


    }









    /*
     * =====================================================
     * VALIDATION FAILURE
     * =====================================================
     */


    return {


        success:false,


        failure:{

            stage:
                "validator",


            reason:

                validation?.reason ||

                "Ответ не прошёл проверку",



            validation


        }


    };


}
