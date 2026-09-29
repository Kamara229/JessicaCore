/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v5
 * =========================================================
 *
 * Финальный исполнитель сохранения Experience Skill.
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
 * - вызывает AI.
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


        learned:false,


        stage,


        error,


        proposal,


        experience:null


    };

}









/*
 * =========================================================
 * SKILL ID
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
 * MAIN
 * =========================================================
 */


export async function approveAndSaveLearningProposal({

    proposal,

    autonomy,

    confidence = null

} = {}) {





    /*
     * INPUT
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
     * AUTONOMY
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
                "Autonomy не разрешил обучение"

        });

    }









    if(
        !proposal.proposedExperience
    ){

        return buildFailure({

            stage:
                "experience",

            proposal,

            error:
                "Нет Experience данных"

        });

    }









    /*
     * SKILL ID
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
                "Skill ID отсутствует"

        });

    }









    /*
     * HISTORY
     */


    let history;



    try{


        history =

            await getExperienceHistory(
                skillId
            );


    }catch(error){


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



    const previousVersion =

        version > 1

        ?

        version - 1

        :

        null;









    /*
     * BUILD
     */


    let experience;



    try{


        experience =

            buildExperienceSkill({

                proposedExperience:

                    proposal.proposedExperience,


                skillId,


                version,


                previousVersion,



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



                metadata:{


                    proposalId:

                        proposal.id,


                    queueItemId:

                        proposal.queueItemId,


                    learnedFrom:

                        proposal.source || 
                        "learning_pipeline",



                    learning:


                    {

                        action:

                            proposal.action,


                        autonomy:


                            autonomy.action,


                        reason:

                            autonomy.reason || ""

                    }


                }


            });



    }catch(error){


        return buildFailure({

            stage:
                "builder",

            proposal,

            error:
                error.message

        });

    }









    /*
     * SAVE
     */


    let saved;



    try{


        saved =

            await saveExperienceSkill(
                experience
            );


    }catch(error){


        return buildFailure({

            stage:
                "storage",

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
                "storage",

            proposal,

            error:
                "Experience Storage не подтвердил сохранение"

        });

    }









    /*
     * APPROVE PROPOSAL
     */


    let approved;



    try{


        approved =

            approveLearningProposal(
                proposal
            );


    }catch(error){


        return {


            success:true,


            learned:true,


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


        learned:true,


        stage:
            "completed",



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
