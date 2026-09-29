/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v3
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
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - принимает решение;
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





function safeString(value){

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
            proposal?.targetSkill?.id
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



    /*
     * INPUT
     */

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







    /*
     * AUTONOMY CHECK
     */

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
                "Нет данных Experience"

        });

    }







    const skillId =

        resolveSkillId(
            proposal
        );



    if(
        !skillId
    ){

        return buildFailure({

            stage:"skill",

            proposal,

            error:
                "Skill ID не определён"

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








    /*
     * BUILD EXPERIENCE
     */


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
                    0


            });


        /*
         * Добавляем данные обучения
         */


        experience.learning = {

            action:

                proposal.action,


            source:

                proposal.source,


            reason:

                proposal.analysis?.reason || "",


            autonomy:

                proposal.autonomy || null


        };


    }catch(error){


        return buildFailure({

            stage:"build",

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

            stage:"save",

            proposal,

            error:
                error.message

        });

    }







    if(
        !saved?.success
    ){

        return buildFailure({

            stage:"save",

            proposal,

            error:
                "Experience Storage не подтвердил сохранение"

        });

    }







    /*
     * APPROVE
     */


    const approved =

        approveLearningProposal(
            proposal
        );






    return {


        success:true,


        stage:"learned",


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
