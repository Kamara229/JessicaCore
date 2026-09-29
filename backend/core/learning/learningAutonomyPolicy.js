/*
 * =========================================================
 * JESSICA LEARNING AUTONOMY POLICY
 * =========================================================
 *
 * Автоматическая политика обучения Jessica.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Autonomy Policy
 *        ↓
 *
 * AUTO_APPROVE
 *
 * или
 *
 * KEEP_CANDIDATE
 *
 *
 * Отвечает только за:
 *
 * - оценку качества обучения;
 * - решение об автоматическом добавлении.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - меняет Experience;
 * - вызывает AI;
 * - работает с БД.
 *
 * =========================================================
 */





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


/*
 * Минимальная уверенность
 * для автоматического обучения.
 */

const MIN_CONFIDENCE =
    0.85;



/*
 * Минимальное количество
 * подтверждений опыта.
 */

const MIN_EXAMPLES =
    3;





/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function normalizeNumber(
    value
) {


    const number =
        Number(value);



    if (
        Number.isNaN(number)
    ) {

        return 0;

    }


    return number;

}







function getExamplesCount(
    proposal
) {


    const examples =
        proposal
            ?.proposedExperience
            ?.examples;



    if (
        !Array.isArray(examples)
    ) {

        return 0;

    }



    return examples.length;

}









/*
 * =========================================================
 * CHECK REUSABLE
 * =========================================================
 */


function isReusable(
    proposal
) {


    return (

        proposal
        ?.analysis
        ?.reusable === true

    );

}









/*
 * =========================================================
 * MAIN POLICY
 * =========================================================
 */


export function evaluateLearningAutonomy(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {


        return {

            action:
                "KEEP_CANDIDATE",


            reason:
                "Proposal отсутствует"

        };

    }






    const confidence =

        normalizeNumber(

            proposal.confidence

        );





    const examples =

        getExamplesCount(
            proposal
        );








    /*
     * =====================================================
     * HARD RULES
     * =====================================================
     */


    if (
        !isReusable(
            proposal
        )
    ) {


        return {

            action:
                "KEEP_CANDIDATE",


            reason:
                "Опыт не признан повторяемым"

        };

    }






    if (
        confidence < MIN_CONFIDENCE
    ) {


        return {

            action:
                "KEEP_CANDIDATE",


            reason:
                "Недостаточная уверенность"

        };

    }







    if (
        examples < MIN_EXAMPLES
    ) {


        return {

            action:
                "KEEP_CANDIDATE",


            reason:
                "Недостаточно примеров успешного применения"

        };

    }







    /*
     * =====================================================
     * AUTO LEARNING
     * =====================================================
     */


    return {


        action:
            "AUTO_APPROVE",



        reason:
            "Опыт соответствует условиям автономного обучения",



        confidence,


        examples


    };


}
