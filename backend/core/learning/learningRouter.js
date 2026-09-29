/*
 * =========================================================
 * JESSICA LEARNING ROUTER v3
 * =========================================================
 *
 * Преобразует результат Experience Analyzer
 * в Learning Event.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Experience Analyzer
 *        ↓
 * Learning Router
 *        ↓
 * Learning Queue
 *        ↓
 * Learning Proposal
 *
 *
 * Ответственность:
 *
 * - создать Learning Event;
 * - сохранить кандидата обучения;
 * - подготовить данные для Proposal.
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - принимает Approval.
 *
 * =========================================================
 */



/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeNumber(
    value
){

    const number =
        Number(value);


    return Number.isFinite(number)
        ?
        number
        :
        0;

}






function normalizeObject(
    value
){

    if(
        !value ||
        typeof value !== "object"
    ){

        return {};

    }


    return value;

}









/*
 * =========================================================
 * IGNORE
 * =========================================================
 */


function createIgnoreEvent(
    reason
){

    return {


        action:

            "IGNORE",



        reason:

            reason ||

            "Обучение не требуется",



        createdAt:

            new Date()
                .toISOString()


    };

}









/*
 * =========================================================
 * BASE EVENT
 * =========================================================
 */


function createBaseEvent(
    analysis
){

    const candidate =

        normalizeObject(

            analysis?.skillCandidate

        );





    return {


        confidence:

            normalizeNumber(

                candidate.confidence

            ),





        reusable:

            analysis?.reusable === true,





        reason:

            analysis?.reason || "",






        payload:

        {


            skillCandidate:

                candidate,



            source:

                "experience-analyzer"



        },




        createdAt:

            new Date()
                .toISOString()


    };

}









/*
 * =========================================================
 * NEW SKILL EVENT
 * =========================================================
 */


function createNewSkillEvent(
    analysis
){

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
 * SKILL IMPROVEMENT EVENT
 * =========================================================
 */


function createSkillImprovementEvent(
    analysis
){

    const base =

        createBaseEvent(
            analysis
        );




    return {


        action:

            "SKILL_IMPROVEMENT",



        ...base,



        payload:

        {


            ...base.payload,



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







    switch(
        analysis.action
    ){



        case "NEW_SKILL":


            return createNewSkillEvent(

                analysis

            );






        case "SKILL_IMPROVEMENT":


            return createSkillImprovementEvent(

                analysis

            );







        case "IGNORE":


            return createIgnoreEvent(

                analysis.reason

            );







        default:


            return createIgnoreEvent(

                "Неизвестный тип Learning Action"

            );


    }


}
