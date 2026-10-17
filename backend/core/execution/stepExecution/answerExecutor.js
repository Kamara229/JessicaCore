/*
 * =========================================================
 * JESSICA EXECUTION
 * ANSWER EXECUTOR v4
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
 * КРИТИЧЕСКО:
 *
 * ошибка Composer является Presentation Failure,
 * а не ошибкой Execution Route.
 *
 *
 * Поэтому:
 *
 * composer failure
 *      ≠
 * tool failure
 *
 * composer failure
 *      ≠
 * validation failure
 *
 * composer failure
 *      ≠
 * reason for replan
 *
 * =========================================================
 */


import {
    composeAnswer
} from "../../answerComposer.js";


/*
 * =========================================================
 * CONSTANTS
 * =========================================================
 */


const PRESENTATION_CATEGORY =
    "presentation";


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

        success:
            false,

        failure: {

            stage,

            failureType,

            category:
                PRESENTATION_CATEGORY,

            reason,

            shouldRetry:
                false,

            needsClarification:
                false,

            noVerifiedResult:
                false,

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

            answerResult?.stage

            ||

            "composer",


        failureType:

            answerResult?.failureType

            ||

            "composer-failure",


        /*
         * КРИТИЧЕСКО:
         *
         * это Presentation Failure.
         *
         * Execution route уже построен
         * и повторно планировать его
         * только из-за Composer нельзя.
         */


        category:
            PRESENTATION_CATEGORY,


        reason:

            answerResult?.reason

            ||

            answerResult?.text

            ||

            "Ответ не создан",


        /*
         * AI technical retry уже выполняется
         * внутри composerRequest → aiRetry.
         *
         * Execution Retry здесь запрещён.
         */


        shouldRetry:
            false,


        needsClarification:

            answerResult
                ?.needsClarification
            ===
            true,


        noVerifiedResult:
            false,


        answerResult

    };

}


/*
 * =========================================================
 * EXECUTE ANSWER
 * =========================================================
 */


export async function executeAnswer(
    context,
    options = {}
) {

    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    if(
        !context
    ){

        return buildFailure(

            "composer",

            "missing-context",

            "Execution context отсутствует"

        );

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

                context.runResult,

                options

            );


    }catch(error){


        return buildFailure(

            "composer",

            "composer-exception",

            error?.message

            ||

            "Ошибка создания ответа",

            {

                error:
                    error || null

            }

        );

    }


    /*
     * =====================================================
     * COMPOSER FAILURE
     * =====================================================
     */


    if(
        context.answerResult?.success
        !==
        true
    ){

        return {

            success:
                false,

            failure:

                normalizeAnswerFailure(
                    context.answerResult
                )

        };

    }


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        answerResult:
            context.answerResult

    };

}
