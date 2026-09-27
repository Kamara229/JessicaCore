/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v2
 * =========================================================
 *
 * Финальный исполнитель подтверждения Learning Proposal.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Resolve Action
 *        ↓
 * Resolve Skill
 *        ↓
 * Get History
 *        ↓
 * Calculate Version
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
 * - вызывает AI;
 * - работает напрямую с БД;
 * - создаёт Proposal.
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
) {

    return String(
        value || ""
    )
    .trim();

}









function buildFailure({

    stage = "approval",

    error = "Ошибка обучения",

    proposal = null,

    skillId = null,

    version = null

} = {}) {


    return {

        success:false,

        stage,

        proposal,

        experience:null,

        skillId,

        version,

        error

    };

}









function normalizeSkillId(
    value
) {

    return safeString(
        value
    );

}









/*
 * =========================================================
 * VERSION
 * =========================================================
 */


function extractVersion(
    item
) {


    const version =

        Number(

            item?.version

            ??

            item?.payload?.version

        );



    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return null;

    }



    return version;

}









function getNextVersion(
    history
) {


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
                extractVersion
            )

            .filter(
                Boolean
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
 * ACTION
 * =========================================================
 */


function resolveAction(
    proposal
) {


    return (

        proposal?.action

        ||

        "NEW_SKILL"

    );

}









/*
 * =========================================================
 * TARGET SKILL
 * =========================================================
 */


function resolveSkillId({

    proposal,

    skillId

}) {


    const experience =
        proposal?.proposedExperience || {};



    const target =
        proposal?.targetSkill || {};



    return (

        normalizeSkillId(
            skillId
        )

        ||

        normalizeSkillId(
            target.id
        )

        ||

        normalizeSkillId(
            experience.id
        )

        ||

        normalizeSkillId(
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
 * MAIN
 * =========================================================
 */


export async function approveAndSaveLearningProposal({

    proposal,

    skillId = "",

    confidence = 0.7

} = {}) {


    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return buildFailure({

            stage:"input",

            error:
                "Learning Proposal отсутствует"

        });

    }





    if(
        !proposal.proposedExperience
    ){

        return buildFailure({

            stage:"input",

            proposal,

            error:
                "Нет proposedExperience"

        });

    }









    const action =
        resolveAction(
            proposal
        );



    const resolvedSkillId =

        resolveSkillId({

            proposal,

            skillId

        });





    if(
        !resolvedSkillId
    ){

        return buildFailure({

            stage:"skill",

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



    try{


        history =

            await getExperienceHistory(
                resolvedSkillId
            );


    }catch(error){


        return buildFailure({

            stage:"history",

            proposal,

            skillId:
                resolvedSkillId,

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



    try{


        experience =

            buildExperienceSkill({

                proposedExperience:

                    proposal.proposedExperience,


                skillId:

                    resolvedSkillId,


                version,


                confidence


            });



    }catch(error){


        return buildFailure({

            stage:"build",

            proposal,

            skillId:
                resolvedSkillId,

            version,

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



    try{


        saved =

            await saveExperienceSkill(
                experience
            );


    }catch(error){


        return buildFailure({

            stage:"save",

            proposal,

            skillId:
                resolvedSkillId,

            version,

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

            skillId:
                resolvedSkillId,

            version,

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



    try{


        approved =

            approveLearningProposal(
                proposal
            );


    }catch(error){


        return {


            success:true,

            stage:"saved",

            proposal,

            experience,

            skillId:
                resolvedSkillId,

            version,

            proposalStateUpdated:false,

            error:
                "Skill сохранён, Proposal не обновлён"


        };

    }









    return {


        success:true,

        stage:"approved",


        proposal:
            approved,


        proposalStateUpdated:
            true,


        experience,


        skillId:
            resolvedSkillId,


        version,


        action,


        previousVersions:

            Array.isArray(history)

                ? history.length

                : 0,


        error:""


    };


}
