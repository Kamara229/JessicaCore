/*
 * =========================================================
 * JESSICA BASIC RESULT VALIDATOR
 * =========================================================
 *
 * Техническая проверка результата.
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - проверяет смысл ответа;
 * - проверяет источники;
 * - анализирует claims.
 *
 *
 * Проверяет только:
 *
 * - существует ли результат;
 * - можно ли показать ответ пользователю;
 * - есть ли техническая ошибка выполнения.
 *
 * =========================================================
 */



/*
 * =========================================================
 * VALIDATE ANSWER
 * =========================================================
 */


function validateAnswer(
    answerResult
) {


    if (
        !answerResult ||
        typeof answerResult !== "object"
    ) {

        return {

            valid:
                false,

            shouldRetry:
                true,

            needsClarification:
                false,

            reason:
                "Ответ отсутствует"

        };

    }



    if (
        answerResult.success !== true
    ) {

        return {

            valid:
                false,

            shouldRetry:
                answerResult.shouldRetry === true,

            needsClarification:
                answerResult.needsClarification === true,

            reason:
                answerResult.text ||
                "Ответ не сформирован"

        };

    }



    if (
        typeof answerResult.text !== "string" ||
        !answerResult.text.trim()
    ) {

        return {

            valid:
                false,

            shouldRetry:
                true,

            needsClarification:
                false,

            reason:
                "Итоговый ответ пустой"

        };

    }


    return null;

}



/*
 * =========================================================
 * VALIDATE EXECUTION
 * =========================================================
 */


function validateExecution(
    taskRunResult
) {


    if (
        !taskRunResult ||
        typeof taskRunResult !== "object"
    ) {

        return null;

    }



    if (
        taskRunResult.success === false
    ) {


        const needsClarification =
            taskRunResult.needsClarification === true;



        return {

            valid:
                false,


            shouldRetry:
                !needsClarification,


            needsClarification,


            reason:
                taskRunResult.text ||
                "Ошибка выполнения задачи"

        };

    }


    return null;

}



/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function validateBasicResult(

    taskRunResult,

    answerResult

) {


    const answerValidation =
        validateAnswer(
            answerResult
        );


    if (
        answerValidation
    ) {

        return answerValidation;

    }



    const executionValidation =
        validateExecution(
            taskRunResult
        );


    if (
        executionValidation
    ) {

        return executionValidation;

    }



    return {

        valid:
            true,

        shouldRetry:
            false,

        needsClarification:
            false,

        reason:
            "Базовые технические проверки пройдены"

    };

}
