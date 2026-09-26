/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL BUILDER
 * =========================================================
 *
 * Создаёт предложение нового знания
 * на основе анализа Execution Trace.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - пишет в Supabase;
 * - изменяет Experience.
 *
 * Только создаёт Proposal.
 *
 * =========================================================
 */


function createProposalId() {

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
 * BUILD PROPOSAL
 * =========================================================
 */


export function buildLearningProposal(
    trace,
    analysis
) {


    if (
        !trace ||
        !analysis
    ) {

        return {

            success:false,

            proposal:null,

            reason:
                "Недостаточно данных для создания Proposal"

        };

    }



    if (
        analysis.reusable !== true ||
        !analysis.skillCandidate
    ) {

        return {

            success:false,

            proposal:null,

            reason:
                "Нет кандидата на обучение"

        };

    }



    const candidate =
        analysis.skillCandidate;



    /*
     * =====================================================
     * PROPOSAL
     * =====================================================
     */


    const proposal = {


        id:
            createProposalId(),



        status:
            "PENDING_APPROVAL",



        skillId:
            candidate.skillId,



        category:
            candidate.category || "unknown",



        confidence:
            candidate.confidence || 0,



        source:
            "execution-learning",



        createdAt:
            new Date()
                .toISOString(),



        example: {


            task:
                trace.task,



            resultStatus:
                trace.status,



            usedTools:
                trace.usedTools || [],



            stats:
                trace.stats || {}

        }



    };



    return {


        success:true,


        readyForApproval:true,


        proposal


    };


}
