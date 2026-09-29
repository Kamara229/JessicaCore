/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v4
 * =========================================================
 *
 * Финальный исполнитель автономного обучения.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Autonomy Decision
 *        ↓
 * Build Experience Skill
 *        ↓
 * Save Experience
 *        ↓
 * Approve Proposal
 *
 *
 * НЕ:
 *
 * - анализирует опыт;
 * - принимает решение обучения;
 * - вызывает AI;
 * - ищет кандидатов.
 *
 * Решение принимает:
 *
 * learningAutonomyPolicy
 *
 * =========================================================
 */



import {
    approveLearningProposal
} from "./learningProposal.js";


import {
    buildExperienceSkill,
    buildLearningSkillId
} from "./learningSkillBuilder.js";


import {
    getExperienceHistory,
    saveExperienceSkill
} from "../storage/experienceStorage.js";









/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function safeString(
    value
){

    return String(
        value || ""
    )
    .trim();

}








function buildFailure({

    stage,

    error,

    proposal = null

} = {}){


    return {


        success:false,


        stage,


        error,


        proposal,


        experience:null


    };


}









/*
 * =========================================================
 * RESOLVE SKILL ID
 * =========================================================
 */


function resolveSkillId(
    proposal
){

    const experience =

        proposal?.proposedExperience || {};




    return (

        safeString(
            proposal?.targetSkill?.id
        )


        ||


        safeString(
            experience.id
        )


        ||


        safeString(
            experience.skillId
        )


        ||


        buildLearningSkillId(
            experience.name
        )

    );


}









/*
 * =========================================================
 * VERSION
 * =========================================================
 */


function getNextVersion(
    history
){

    if(
        !Array.isArray(history)
        ||
        history.length === 0
    ){

        return 1;

    }



    const versions =

        history

            .map(

                item =>

                    Number(

                        item.version

                        ||

                        item.payload?.version

                    )

            )

            .filter(

                value =>

                    Number.isInteger(value)

                    &&

                    value > 0

            );




    if(
        versions.length === 0
    ){

        return 1;

    }



    return Math.max(
        ...versions
    ) + 1;


}









/*
 * =========================================================
 * MAIN APPROVAL
 * =========================================================
 */


export async function approveAndSaveLearningProposal({

    proposal,

    autonomy,

    confidence = null

} = {}) {



    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return buildFailure({

            stage:
                "input",

            error:
                "Proposal отсутствует"

        });

    }









    /*
     * =====================================================
     * AUTONOMY RESULT
     * =====================================================
     */


    if(
        !autonomy
        ||
        autonomy.action !==
        "AUTO_APPROVE"
    ){

        return buildFailure({

            stage:
                "autonomy",

            proposal,

            error:
                "Proposal не разрешён для автоматического обучения"

        });

    }









    /*
     * =====================================================
     * EXPERIENCE DATA
     * =====================================================
     */


    if(
        !proposal.proposedExperience
    ){

        return buildFailure({

            stage:
                "experience",

            proposal,

            error:
                "Нет данных Experience"

        });

    }









    /*
     * =====================================================
     * SKILL ID
     * =====================================================
     */


    const skillId =

        resolveSkillId(
            proposal
        );




    if(
        !skillId
    ){

        return buildFailure({

            stage:
                "skill",

            proposal,

            error:
                "Skill ID не определён"

        });

    }









    /*
     * =====================================================
     * HISTORY
     * =====================================================
     */


    let history;



    try {


        history =

            await getExperienceHistory(
                skillId
            );


    } catch(error){


        return buildFailure({

            stage:
                "history",

            proposal,

            error:
                error.message

        });


    }









    const version =

        getNextVersion(
            history
        );









    /*
     * =====================================================
     * BUILD EXPERIENCE
     * =====================================================
     */


    let experience;



    try {


        experience =

            buildExperienceSkill({

                proposedExperience:

                    proposal.proposedExperience,


                skillId,


                version,


                previousVersion:

                    version > 1

                    ?

                    version - 1

                    :

                    null,



                mode:

                    proposal.action ===
                    "SKILL_IMPROVEMENT"

                    ?

                    "update"

                    :

                    "create",




                confidence:

                    confidence ??

                    proposal.confidence ??

                    0.7,



                metadata:

                {

                    proposalId:

                        proposal.id,



                    queueItemId:

                        proposal.queueItemId,



                    learnedFrom:

                        proposal.source || 
                        "learning_pipeline"


                }


            });



    }catch(error){


        return buildFailure({

            stage:
                "build",

            proposal,

            error:
                error.message

        });


    }









    /*
     * =====================================================
     * SAVE
     * =====================================================
     */


    let saved;



    try {


        saved =

            await saveExperienceSkill(
                experience
            );


    }catch(error){


        return buildFailure({

            stage:
                "save",

            proposal,

            error:
                error.message

        });


    }









    if(
        !saved?.success
    ){

        return buildFailure({

            stage:
                "save",

            proposal,

            error:
                "Experience Storage не подтвердил сохранение"

        });

    }









    /*
     * =====================================================
     * APPROVE PROPOSAL
     * =====================================================
     */


    let approved;



    try {


        approved =

            approveLearningProposal(
                proposal
            );


    }catch(error){


        return {


            success:true,


            stage:
                "saved",


            proposal,


            experience,


            skillId,


            version,


            proposalStateUpdated:false,


            error:
                error.message


        };


    }









    return {


        success:true,


        stage:
            "learned",



        proposal:
            approved,



        experience,



        skillId,



        version,



        action:
            proposal.action,



        previousVersions:

            history.length || 0



    };


}
