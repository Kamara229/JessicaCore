/*
 * =========================================================
 * JESSICA LEARNING AUTONOMY POLICY v2
 * =========================================================
 *
 * Автоматическое решение:
 *
 * добавлять Experience Skill
 * или оставить кандидатом.
 *
 *
 * Цель:
 *
 * Jessica должна обучаться автоматически
 * на основе успешных действий пользователя.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - пишет в БД;
 * - вызывает AI.
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
 * для нового Skill.
 */

const NEW_SKILL_MIN_CONFIDENCE =
    0.70;



/*
 * Минимальная уверенность
 * для улучшения существующего.
 */

const IMPROVEMENT_MIN_CONFIDENCE =
    0.60;





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






function getExamplesCount(
    proposal
) {


    const examples =

        proposal
            ?.proposedExperience
            ?.examples;



    if(
        !Array.isArray(examples)
    ){

        return 0;

    }


    return examples.length;

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







function hasExperience(
    proposal
) {


    return (

        proposal
            ?.proposedExperience
            ?.workflow
            ?.length > 0

        ||

        proposal
            ?.proposedExperience
            ?.examples
            ?.length > 0

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








    /*
     * =====================================================
     * COMMON CHECKS
     * =====================================================
     */


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
        !hasExperience(
            proposal
        )
    ){

        return {


            action:
                "KEEP_CANDIDATE",


            reason:
                "Недостаточно структуры Skill"


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
                    "Новый Skill имеет достаточную уверенность",


                confidence,


                examples


            };

        }




        return {


            action:
                "KEEP_CANDIDATE",


            reason:
                "Новый Skill требует дополнительного опыта",


            confidence,


            examples


        };


    }









    /*
     * =====================================================
     * SKILL IMPROVEMENT
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
                    "Существующий Skill получил успешное улучшение",


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








    /*
     * =====================================================
     * UNKNOWN
     * =====================================================
     */


    return {


        action:
            "KEEP_CANDIDATE",


        reason:
            "Неизвестный тип обучения"


    };


}
