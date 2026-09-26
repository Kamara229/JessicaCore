/*
 * =========================================================
 * JESSICA EXPERIENCE MEMORY ADAPTER
 * =========================================================
 *
 * Преобразует подтверждённый Learning Proposal
 * в объект Experience.
 *
 *
 * НЕ:
 *
 * - спрашивает пользователя;
 * - принимает решение об обучении;
 * - вызывает AI;
 * - делает approve.
 *
 *
 * Только адаптация структуры.
 *
 * =========================================================
 */



/*
 * =========================================================
 * BUILD EXPERIENCE RECORD
 * =========================================================
 */


export function buildExperienceRecord(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {

        return {

            success:false,

            experience:null,

            reason:
                "Proposal отсутствует"

        };

    }



    if (
        proposal.status !==
        "PENDING_APPROVAL" &&
        proposal.status !==
        "APPROVED"
    ) {

        return {

            success:false,

            experience:null,

            reason:
                "Proposal не готов для Experience"

        };

    }




    /*
     * =====================================================
     * EXPERIENCE OBJECT
     * =====================================================
     */


    const experience = {


        skillId:
            proposal.skillId,



        category:
            proposal.category ||
            "general",



        version:
            1,



        confidence:
            proposal.confidence ||
            0.5,



        source:
            proposal.source ||
            "learning",



        examples:[

            {

                task:
                    proposal.example?.task ||
                    "",


                resultStatus:
                    proposal.example?.resultStatus ||
                    "",


                usedTools:
                    proposal.example?.usedTools ||
                    []

            }

        ],



        metadata: {


            createdBy:
                "jessica-learning",



            createdAt:
                new Date()
                    .toISOString()



        }


    };




    return {


        success:true,


        experience


    };


}
