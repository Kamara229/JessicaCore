/*
 * =========================================================
 * JESSICA LEARNING APPROVAL
 * =========================================================
 *
 * Финальный этап обучения Jessica.
 *
 *
 * Поддерживает:
 *
 * NEW_SKILL
 *      ↓
 * создание нового Experience Skill
 *
 *
 * SKILL_IMPROVEMENT
 *      ↓
 * создание новой версии существующего Skill
 *
 *
 * Flow:
 *
 * Proposal
 *      ↓
 * Resolve Action
 *      ↓
 * Resolve Target Skill
 *      ↓
 * History
 *      ↓
 * Next Version
 *      ↓
 * Build Experience
 *      ↓
 * Atomic Save
 *      ↓
 * Approve Proposal
 *
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - принимает решение;
 * - вызывает AI;
 * - работает напрямую с Supabase.
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
 * NORMALIZE
 * =========================================================
 */


function normalizeSkillId(
    value
) {

    return String(
        value || ""
    )
    .trim();

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
            item?.version ??
            item?.payload?.version
        );



    if (
        !Number.isInteger(version) ||
        version < 1
    ) {

        return null;

    }


    return version;

}





function getNextVersion(
    history
) {


    if (
        !Array.isArray(history) ||
        history.length === 0
    ) {

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



    if (
        versions.length === 0
    ) {

        return 1;

    }



    return (

        Math.max(
            ...versions
        )

        +

        1

    );

}









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure({

    stage,

    error,

    proposal = null,

    skillId = null,

    version = null

} = {}) {


    return {


        success:false,


        stage:


            stage || "approval",



        proposal,



        experience:null,



        skillId,



        version,



        error:

            error ||

            "Ошибка обучения"


    };

}









/*
 * =========================================================
 * RESOLVE ACTION
 * =========================================================
 */


function resolveAction(
    proposal,
    decision
) {


    return (

        decision?.action

        ||

        proposal?.action

        ||

        "NEW_SKILL"

    );

}









/*
 * =========================================================
 * RESOLVE TARGET
 * =========================================================
 */


function resolveTargetSkill({

    action,

    proposal,

    skillId

}) {


    const experience =

        proposal?.proposedExperience || {};



    const target =

        proposal?.targetSkill || {};






    /*
     * Новый Skill
     */


    if (
        action === "NEW_SKILL"
    ) {


        return {


            mode:

                "create",



            skillId:


                normalizeSkillId(
                    skillId
                )


                ||

                normalizeSkillId(
                    experience.id ||
                    experience.skillId
                )


                ||

                buildLearningSkillId(
                    experience.name
                ),



            previousVersion:

                null


        };

    }








    /*
     * Улучшение Skill
     */


    if (
        action === "SKILL_IMPROVEMENT"
    ) {


        return {


            mode:

                "update",



            skillId:


                normalizeSkillId(
                    skillId
                )


                ||

                normalizeSkillId(
                    target.id
                ),



            previousVersion:

                Number(
                    target.version || 0
                )


        };

    }








    return {


        mode:

            "create",



        skillId:


            normalizeSkillId(
                experience.id ||
                experience.skillId
            )


            ||

            buildLearningSkillId(
                experience.name
            ),



        previousVersion:

            null


    };


}









/*
 * =========================================================
 * MAIN APPROVAL
 * =========================================================
 */


export async function approveAndSaveLearning({

    proposal,

    decision = null,

    skillId = "",

    confidence = 0.7

} = {}) {



    /*
     * =====================================================
     * VALIDATE INPUT
     * =====================================================
     */


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {


        return buildFailure({

            stage:
                "input",

            error:
                "Learning Proposal отсутствует"

        });

    }






    if (
        !proposal.proposedExperience
    ) {


        return buildFailure({

            stage:
                "input",

            proposal,

            error:
                "Нет proposedExperience"

        });

    }








    /*
     * =====================================================
     * ACTION
     * =====================================================
     */


    const action =

        resolveAction(
            proposal,
            decision
        );









    /*
     * =====================================================
     * TARGET
     * =====================================================
     */


    const target =

        resolveTargetSkill({

            action,

            proposal,

            skillId

        });





    if (
        !target.skillId
    ) {


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
                target.skillId
            );


    } catch(error) {


        return buildFailure({

            stage:
                "history",

            proposal,

            skillId:
                target.skillId,

            error:
                error.message

        });

    }








    /*
     * =====================================================
     * VERSION
     * =====================================================
     */


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


                skillId:

                    target.skillId,


                version,


                confidence

            });



    } catch(error) {


        return buildFailure({

            stage:
                "build",

            proposal,

            skillId:
                target.skillId,

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


    let saveResult;



    try {


        saveResult =

            await saveExperienceSkill(
                experience
            );



    } catch(error) {


        return buildFailure({

            stage:
                "save",

            proposal,

            skillId:
                target.skillId,

            version,

            error:
                error.message

        });

    }








    if (
        !saveResult?.success
    ) {


        return buildFailure({

            stage:
                "save",

            proposal,

            skillId:
                target.skillId,

            version,

            error:
                "Experience Storage не подтвердил сохранение"

        });

    }









    /*
     * =====================================================
     * APPROVE
     * =====================================================
     */


    let approvedProposal;



    try {


        approvedProposal =

            approveLearningProposal(
                proposal
            );



    } catch(error) {


        return {


            success:true,


            stage:
                "saved",


            proposal,


            experience,


            skillId:
                target.skillId,


            version,


            proposalStateUpdated:false,


            error:
                "Skill сохранён, но Proposal не обновлён"


        };


    }









    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {


        success:true,


        stage:
            "approved",



        proposal:

            approvedProposal,



        proposalStateUpdated:

            true,



        experience,



        skillId:

            target.skillId,



        version,



        mode:

            target.mode,



        previousVersions:

            Array.isArray(history)

                ? history.length

                : 0,



        error:""


    };


}
