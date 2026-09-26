/*
 * =========================================================
 * JESSICA FEEDBACK MEMORY ADAPTER
 * =========================================================
 *
 * Преобразует Feedback от Jessica
 * в предложение улучшения Experience.
 *
 *
 * НЕ:
 *
 * - сохраняет в Supabase;
 * - меняет Skill напрямую;
 * - принимает решение об обучении.
 *
 *
 * Только готовит Improvement Proposal.
 *
 * =========================================================
 */



/*
 * =========================================================
 * CREATE ID
 * =========================================================
 */


function createId() {


    try {


        if (
            typeof crypto !== "undefined" &&
            crypto.randomUUID
        ) {

            return crypto.randomUUID();

        }


    } catch(error) {

    }



    return (

        Date.now()
        +
        "-"
        +
        Math.random()
            .toString(36)
            .substring(2)

    );


}





/*
 * =========================================================
 * BUILD IMPROVEMENT
 * =========================================================
 */


export function buildFeedbackImprovementProposal(
    feedback
) {


    if (
        !feedback ||
        typeof feedback !== "object"
    ) {

        return {

            success:false,

            proposal:null,

            reason:
                "Feedback отсутствует"

        };

    }




    if (
        feedback.reusable !== true &&
        !feedback.feedback
    ) {

        return {

            success:false,

            proposal:null,

            reason:
                "Нет данных для улучшения"

        };

    }



    const data =
        feedback.feedback;




    if (
        !data?.lesson
    ) {

        return {

            success:false,

            proposal:null,

            reason:
                "Нет правила улучшения"

        };

    }





    /*
     * =====================================================
     * IMPROVEMENT PROPOSAL
     * =====================================================
     */


    const proposal = {


        id:
            createId(),



        status:
            "PENDING_APPROVAL",



        type:
            "SKILL_IMPROVEMENT",



        skillId:
            data.skillId || null,



        source:
            "execution-feedback",



        priority:
            data.priority || "medium",



        improvement:
        {

            rule:
                data.lesson,



            errorType:
                data.errorType,



            exampleTask:
                data.task

        },



        createdAt:
            new Date()
                .toISOString()



    };





    return {


        success:true,


        readyForApproval:true,


        proposal


    };


}
