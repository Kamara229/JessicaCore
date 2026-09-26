/*
 * =========================================================
 * JESSICA LEARNING ROUTER
 * =========================================================
 *
 * Маршрутизатор Learning Pipeline.
 *
 *
 * Получает результат Analyzer:
 *
 * Experience Analyzer
 *        ↓
 * Learning Router
 *        ↓
 * Learning Event
 *
 *
 * Возможные действия:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * IGNORE
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - вызывает Supabase;
 * - создаёт Skill;
 * - делает Approval.
 *
 * =========================================================
 */






/*
 * =========================================================
 * IGNORE EVENT
 * =========================================================
 */


function createIgnoreEvent(
    reason
) {


    return {


        action:

            "IGNORE",



        reason:

            reason ||

            "Обучение не требуется"

    };

}









/*
 * =========================================================
 * NEW SKILL
 * =========================================================
 */


function createNewSkillEvent(
    analysis
) {


    return {


        action:

            "NEW_SKILL",



        confidence:

            analysis
                ?.skillCandidate
                ?.confidence || 0,



        payload:

            {


                skillCandidate:

                    analysis.skillCandidate,



                source:

                    "experience-analyzer"


            }



    };

}









/*
 * =========================================================
 * SKILL IMPROVEMENT
 * =========================================================
 */


function createSkillImprovementEvent(
    analysis
) {


    return {


        action:

            "SKILL_IMPROVEMENT",



        confidence:

            analysis
                ?.skillCandidate
                ?.confidence || 0,



        payload:

            {


                skills:

                    analysis
                        .skillCandidate
                        ?.skills || [],



                source:

                    "experience-analyzer"


            }



    };

}









/*
 * =========================================================
 * ROUTER
 * =========================================================
 */


export function routeLearningEvent({

    analysis = null

} = {}) {


    if (
        !analysis ||
        typeof analysis !== "object"
    ) {


        return createIgnoreEvent(

            "Нет результата анализа"

        );

    }








    /*
     * =====================================================
     * IGNORE
     * =====================================================
     */


    if (
        analysis.action === "IGNORE"
    ) {


        return createIgnoreEvent(

            analysis.reason

        );

    }









    /*
     * =====================================================
     * NEW SKILL
     * =====================================================
     */


    if (
        analysis.action === "NEW_SKILL"
    ) {


        return createNewSkillEvent(

            analysis

        );

    }









    /*
     * =====================================================
     * SKILL IMPROVEMENT
     * =====================================================
     */


    if (
        analysis.action === "SKILL_IMPROVEMENT"
    ) {


        return createSkillImprovementEvent(

            analysis

        );

    }









    /*
     * =====================================================
     * UNKNOWN
     * =====================================================
     */


    return createIgnoreEvent(

        "Неизвестный тип Learning Action"

    );


}
