/*
 * =========================================================
 * JESSICA LEARNING ROUTER
 * =========================================================
 *
 * Центральный маршрутизатор обучения Jessica.
 *
 *
 * Определяет тип обучения:
 *
 * NEW_SKILL
 * SKILL_IMPROVEMENT
 * IGNORE
 *
 *
 * НЕ:
 *
 * - сохраняет данные;
 * - вызывает Supabase;
 * - подтверждает обучение.
 *
 * =========================================================
 */



/*
 * =========================================================
 * ROUTE LEARNING EVENT
 * =========================================================
 */


export function routeLearningEvent({

    analysis = null,

    proposal = null,

    improvement = null,

    quality = null

} = {}) {



    /*
     * =====================================================
     * QUALITY CHECK
     * =====================================================
     */


    if (
        quality &&
        quality.approved === false
    ) {

        return {

            action:
                "IGNORE",


            reason:
                quality.reason || 
                "Опыт не прошёл проверку качества"


        };

    }




    /*
     * =====================================================
     * NEW SKILL
     * =====================================================
     */


    if (
        proposal?.success === true &&
        proposal.proposal
    ) {


        return {

            action:
                "NEW_SKILL",


            payload:
                proposal.proposal


        };

    }





    /*
     * =====================================================
     * SKILL IMPROVEMENT
     * =====================================================
     */


    if (
        improvement?.success === true &&
        improvement.proposal
    ) {


        return {

            action:
                "SKILL_IMPROVEMENT",


            payload:
                improvement.proposal


        };

    }





    /*
     * =====================================================
     * NOTHING
     * =====================================================
     */


    return {

        action:
            "IGNORE",


        reason:
            "Нет подходящего обучения"


    };


}
