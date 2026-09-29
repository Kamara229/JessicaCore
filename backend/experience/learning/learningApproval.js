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
 * Autonomy Policy
 *        ↓
 * AUTO_APPROVE
 *        ↓
 * Build Experience Skill
 *        ↓
 * Save Experience
 *        ↓
 * Approve Proposal
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

}){


    return {

        success:false,

        stage,

        error,

        proposal,

        experience:null

    };

}








function resolveSkillId(
    proposal
){

    const experience =
        proposal.proposedExperience || {};



    return (

        safeString(
            proposal.targetSkill?.id
        )

        ||

        safeString(
            experience.id
        )

        ||

        buildLearningSkillId(
            experience.name
        )

    );

}








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
                    item.version ||
                    item.payload?.version
                )

        )

        .filter(

            item =>

                Number.isInteger(item)

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









export async function approveAndSaveLearningProposal({

    proposal,

    confidence = null

} = {}) {



    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return buildFailure({

            stage:"input",

            error:
                "Proposal отсутствует"

        });

    }







    if(
        proposal.autonomy?.action !==
        "AUTO_APPROVE"
    ){

        return buildFailure({

            stage:"autonomy",

            proposal,

            error:
                "Proposal не прошёл автономное обучение"

        });

    }







    if(
        !proposal.proposedExperience
    ){

        return buildFailure({

            stage:"experience",

            proposal,

            error:
                "Нет Experience"

        });

    }







    const skillId =

        resolveSkillId(
            proposal
        );



    let history;


    try{


        history =

            await getExperienceHistory(
                skillId
            );


    }catch(error){


        return buildFailure({

            stage:"history",

            proposal,

            error:
                error.message

        });

    }







    const version =

        getNextVersion(
            history
        );







    let experience;


    try{


        experience =

            buildExperienceSkill({

                proposedExperience:

                    proposal.proposedExperience,


                skillId,


                version,


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
                        proposal.source,


                    learningMode:
                        "autonomous",


                    sourceExperience:
                        proposal.proposedExperience


                }


            });



    }catch(error){


        return buildFailure({

            stage:"builder",

            proposal,

            error:
                error.message

        });

    }







    let saved;


    try{


        saved =

            await saveExperienceSkill(
                experience
            );


    }catch(error){


        return buildFailure({

            stage:"storage",

            proposal,

            error:
                error.message

        });

    }







    if(
        !saved?.success
    ){

        return buildFailure({

            stage:"storage",

            proposal,

            error:
                "Experience не сохранён"

        });

    }








    const approved =

        approveLearningProposal(
            proposal
        );







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

            history.length


    };


}
