/*
 * =========================================================
 * JESSICA EXECUTION
 * ANSWER EXECUTOR v2
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
 * Ответственность:
 *
 * - вызвать Answer Composer;
 * - сохранить результат ответа;
 * - нормализовать ошибку.
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
 * NORMALIZE FAILURE
 * =========================================================
 */


function normalizeAnswerFailure(

    answerResult

) {


    return {


        stage:

            answerResult?.stage ||

            "composer",



        failureType:

            answerResult?.failureType ||

            "composer-failure",



        reason:

            answerResult?.reason ||

            answerResult?.text ||

            "Ответ не создан",



        shouldRetry:

            answerResult?.shouldRetry === true,



        needsClarification:

            answerResult?.needsClarification === true,



        answerResult

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


        return {


            success:false,


            failure:

                normalizeAnswerFailure(

                    context.answerResult

                )


        };


    }









    return {


        success:true,


        answerResult:

            context.answerResult


    };


}
