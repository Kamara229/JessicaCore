/*
 * =========================================================
 * JESSICA LEARNING QUALITY GATE
 * =========================================================
 *
 * Проверяет качество будущего обучения.
 *
 *
 * Решает:
 *
 * - сохранять опыт;
 * - отправить на подтверждение;
 * - отклонить.
 *
 *
 * НЕ:
 *
 * - пишет в Supabase;
 * - создаёт Skill;
 * - вызывает AI.
 *
 * =========================================================
 */



/*
 * =========================================================
 * MINIMUM CONDITIONS
 * =========================================================
 */


const MIN_CONFIDENCE =
    0.6;



const MIN_SUCCESS_COUNT =
    1;




/*
 * =========================================================
 * CHECK TRACE QUALITY
 * =========================================================
 */


function checkExecutionQuality(
    trace
) {


    const completed =
        trace?.stats?.completed || 0;



    if (
        completed < MIN_SUCCESS_COUNT
    ) {

        return {

            passed:false,

            reason:
                "Нет успешных выполнений"

        };

    }



    return {

        passed:true

    };


}





/*
 * =========================================================
 * CHECK SKILL QUALITY
 * =========================================================
 */


function checkSkillCandidate(
    candidate
) {


    if (
        !candidate
    ) {

        return {

            passed:false,

            reason:
                "Нет кандидата Skill"

        };

    }



    if (
        candidate.confidence <
        MIN_CONFIDENCE
    ) {

        return {

            passed:false,

            reason:
                "Недостаточная уверенность"

        };

    }



    return {

        passed:true

    };


}





/*
 * =========================================================
 * QUALITY CHECK
 * =========================================================
 */


export function validateLearningQuality(
    trace,
    analysis
) {


    const execution =
        checkExecutionQuality(
            trace
        );



    if (
        !execution.passed
    ) {

        return {

            approved:false,

            reason:
                execution.reason

        };

    }




    const skill =
        checkSkillCandidate(
            analysis?.skillCandidate
        );



    if (
        !skill.passed
    ) {

        return {

            approved:false,

            reason:
                skill.reason

        };

    }



    return {

        approved:true,

        reason:
            "Опыт прошёл проверку качества"

    };


}
