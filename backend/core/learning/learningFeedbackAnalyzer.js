/*
 * =========================================================
 * JESSICA LEARNING FEEDBACK ANALYZER
 * =========================================================
 *
 * Анализирует ошибки Jessica
 * и превращает их в опыт улучшения.
 *
 *
 * НЕ:
 *
 * - изменяет Skill;
 * - пишет в Supabase;
 * - выполняет повторную задачу.
 *
 * Только создаёт Feedback Proposal.
 *
 * =========================================================
 */



/*
 * =========================================================
 * ERROR TYPES
 * =========================================================
 */


function detectErrorType(
    result
) {


    if (
        !result
    ) {

        return "unknown";

    }



    if (
        result.failureType
    ) {

        return result.failureType;

    }



    if (
        result.status === "FAILED"
    ) {

        return "execution-failure";

    }



    if (
        result.resultType ===
        "no_verified_result"
    ) {

        return "verification-failure";

    }



    return "unknown";

}





/*
 * =========================================================
 * EXTRACT LESSON
 * =========================================================
 */


function extractLesson(
    result
) {


    const errorType =
        detectErrorType(
            result
        );



    switch(
        errorType
    ) {


        case "validation-failure":

            return {

                rule:

                    "Ответ необходимо дополнительно проверять перед выдачей",


                priority:
                    "high"

            };



        case "verification-failure":

            return {

                rule:

                    "Недостаточно найти источник, нужно подтвердить данные",


                priority:
                    "high"

            };



        case "execution-failure":

            return {

                rule:

                    "Текущий план выполнения требует улучшения",


                priority:
                    "medium"

            };



        default:

            return {

                rule:

                    "Необходимо дополнительное наблюдение",


                priority:
                    "low"

            };


    }

}





/*
 * =========================================================
 * ANALYZE FEEDBACK
 * =========================================================
 */


export function analyzeLearningFeedback(
    trace,
    result
) {


    if (
        !trace ||
        !result
    ) {

        return {

            reusable:
                false,

            feedback:
                null

        };

    }



    if (
        result.status ===
        "COMPLETED"
        &&
        result.validated === true
    ) {

        return {

            reusable:
                false,

            reason:
                "Ошибки отсутствуют",

            feedback:
                null

        };

    }




    const lesson =
        extractLesson(
            result
        );



    return {

        reusable:
            true,


        feedback:
        {

            skillId:
                trace?.experience?.skillId ||
                null,


            task:
                trace.task,


            errorType:
                detectErrorType(result),


            lesson:
                lesson.rule,


            priority:
                lesson.priority,


            source:
                "execution-feedback"


        }

    };


}
