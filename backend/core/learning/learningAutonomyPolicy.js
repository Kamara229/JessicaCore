/*
 * =========================================================
 * JESSICA LEARNING AUTONOMY POLICY v3
 * =========================================================
 *
 * Решает:
 *
 * может ли Jessica автоматически
 * добавить новый Experience Skill.
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
 * НЕ:
 *
 * - сохраняет Skill;
 * - работает с БД;
 * - вызывает AI.
 *
 * =========================================================
 */





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const NEW_SKILL_MIN_CONFIDENCE =
    0.80;



const IMPROVEMENT_MIN_CONFIDENCE =
    0.60;



const MIN_SUCCESS_RATE =
    0.80;



const MIN_MATURITY =
    0.50;



const MIN_EXAMPLES =
    1;









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



    if(
        Number.isNaN(number)
    ){

        return 0;

    }


    return number;

}







function getAction(
    proposal
) {

    return (

        proposal.action ||

        "NEW_SKILL"

    );

}







function getExperience(
    proposal
) {

    return (

        proposal
            ?.proposedExperience
        ||

        {}

    );

}







function getExamplesCount(
    proposal
) {


    const examples =

        getExperience(
            proposal
        )
        .examples;



    return Array.isArray(examples)

        ? examples.length

        : 0;


}







function isReusable(
    proposal
) {


    return (

        proposal
            ?.analysis
            ?.reusable === true

    );

}







function hasExperienceStructure(
    proposal
) {


    const experience =
        getExperience(
            proposal
        );


    return (

        experience.workflow?.length > 0

        ||

        experience.examples?.length > 0

    );

}







function getQualityMetrics(
    proposal
) {


    const experience =
        getExperience(
            proposal
        );


    return {


        maturity:

            normalizeNumber(
                experience.maturity
            ),



        occurrences:

            normalizeNumber(
                experience.occurrences
            ),



        successRate:

            normalizeNumber(
                experience.successRate
            )

    };

}









/*
 * =========================================================
 * QUALITY CHECK
 * =========================================================
 */


function validateQuality(
    proposal
) {


    const metrics =
        getQualityMetrics(
            proposal
        );



    /*
     * Если метрики ещё не заполнены,
     * допускаем обучение на первом успешном опыте.
     *
     * Позже они будут приходить
     * из Experience Memory.
     */


    if(
        metrics.successRate > 0 &&
        metrics.successRate < MIN_SUCCESS_RATE
    ){

        return {

            ok:false,

            reason:
                "Низкий показатель успешности"

        };

    }




    if(
        metrics.maturity > 0 &&
        metrics.maturity < MIN_MATURITY
    ){

        return {

            ok:false,

            reason:
                "Недостаточная зрелость Skill"

        };

    }




    return {

        ok:true

    };


}









/*
 * =========================================================
 * MAIN
 * =========================================================
 */


export function evaluateLearningAutonomy(
    proposal
) {


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

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




    const action =

        getAction(
            proposal
        );




    const examples =

        getExamplesCount(
            proposal
        );









    if(
        !isReusable(
            proposal
        )
    ){

        return {

            action:
                "KEEP_CANDIDATE",

            reason:
                "Опыт не признан повторяемым"

        };

    }








    if(
        !hasExperienceStructure(
            proposal
        )
    ){

        return {

            action:
                "KEEP_CANDIDATE",

            reason:
                "Нет структуры Skill"

        };

    }







    if(
        examples <
        MIN_EXAMPLES
    ){

        return {

            action:
                "KEEP_CANDIDATE",

            reason:
                "Нет подтверждённых примеров"

        };

    }








    const quality =
        validateQuality(
            proposal
        );



    if(
        !quality.ok
    ){

        return {

            action:
                "KEEP_CANDIDATE",

            reason:
                quality.reason

        };

    }









    /*
     * =====================================================
     * NEW SKILL
     * =====================================================
     */


    if(
        action ===
        "NEW_SKILL"
    ){


        if(
            confidence >=
            NEW_SKILL_MIN_CONFIDENCE
        ){

            return {


                action:
                    "AUTO_APPROVE",


                mode:
                    "NEW_SKILL",


                reason:
                    "Новый Skill прошёл проверку автономного обучения",


                confidence,


                examples

            };

        }


        return {


            action:
                "KEEP_CANDIDATE",


            reason:
                "Недостаточная уверенность для нового Skill",


            confidence,


            examples


        };


    }









    /*
     * =====================================================
     * IMPROVEMENT
     * =====================================================
     */


    if(
        action ===
        "SKILL_IMPROVEMENT"
    ){


        if(
            confidence >=
            IMPROVEMENT_MIN_CONFIDENCE
        ){


            return {


                action:
                    "AUTO_APPROVE",


                mode:
                    "SKILL_IMPROVEMENT",


                reason:
                    "Существующий Skill может быть улучшен",


                confidence,


                examples


            };

        }




        return {


            action:
                "KEEP_CANDIDATE",


            reason:
                "Недостаточная уверенность для изменения Skill",


            confidence,


            examples


        };


    }








    return {


        action:
            "KEEP_CANDIDATE",


        reason:
            "Неизвестный тип обучения"


    };


}
