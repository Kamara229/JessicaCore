/*
 * =========================================================
 * JESSICA EXECUTION
 * ANSWER EXECUTOR v1
 * =========================================================
 *
 * Создание пользовательского ответа.
 *
 *
 * Flow:
 *
 * Run Result
 *      ↓
 * Answer Composer
 *      ↓
 * Answer Result
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - выполняет Validation.
 *
 * =========================================================
 */


import {
    composeAnswer
} from "../../answerComposer.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure(

    stage,

    failureType,

    reason,

    extra = {}

) {


    return {


        success:false,


        failure:

        {


            stage,


            failureType,


            reason,


            ...extra


        }


    };

}









/*
 * =========================================================
 * EXECUTE ANSWER
 * =========================================================
 */


export async function executeAnswer(

    context

) {


    if (
        !context
    ) {


        return buildFailure(

            "composer",

            "missing-context",

            "Execution context отсутствует"

        );

    }









    try {


        context.answerResult =

            await composeAnswer(

                context.task,

                context.plan,

                context.runResult

            );


    }

    catch(error){


        return buildFailure(

            "composer",

            "composer-exception",

            error?.message ||

            "Ошибка создания ответа"

        );


    }









    if (

        context.answerResult?.success !== true

    ) {


        return buildFailure(

            "composer",

            "composer-failure",

            context.answerResult?.text ||

            "Ответ не создан",


            {

                answerResult:

                    context.answerResult

            }

        );


    }









    return {


        success:true,


        answerResult:

            context.answerResult


    };


}
