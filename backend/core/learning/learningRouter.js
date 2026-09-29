/*
 * =========================================================
 * JESSICA LEARNING ROUTER v2
 * =========================================================
 *
 * Маршрутизатор Learning Pipeline.
 *
 *
 * Flow:
 *
 * Experience Analyzer
 *        ↓
 * Learning Router
 *        ↓
 * Learning Event
 *
 *
 * Ответственность:
 *
 * - преобразовать Analysis в Event;
 * - сохранить данные кандидата;
 * - передать информацию дальше.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - принимает решение обучения.
 *
 * =========================================================
 */







/*
 * =========================================================
 * IGNORE
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
 * CREATE EVENT BASE
 * =========================================================
 */


function createBaseEvent(
    analysis
) {


    const candidate =

        analysis?.skillCandidate || {};




    return {


        confidence:

            Number(
                candidate.confidence || 0
            ),



        reusable:

            analysis.reusable === true,



        reason:

            analysis.reason || "",



        payload:

        {


            skillCandidate:

                candidate,



            source:

                "experience-analyzer"



        }


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



        ...createBaseEvent(
            analysis
        )


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


    const event =

        createBaseEvent(
            analysis
        );





    return {


        action:

            "SKILL_IMPROVEMENT",



        ...event,


        payload:

        {


            ...event.payload,



            skills:

                analysis
                    ?.skillCandidate
                    ?.skills || []

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


    if(
        !analysis ||
        typeof analysis !== "object"
    ){

        return createIgnoreEvent(

            "Нет результата анализа"

        );

    }









    if(
        analysis.action === "IGNORE"
    ){

        return createIgnoreEvent(

            analysis.reason

        );

    }









    if(
        analysis.action === "NEW_SKILL"
    ){

        return createNewSkillEvent(
            analysis
        );

    }









    if(
        analysis.action === "SKILL_IMPROVEMENT"
    ){

        return createSkillImprovementEvent(
            analysis
        );

    }









    return createIgnoreEvent(

        "Неизвестный тип Learning Action"

    );


}
