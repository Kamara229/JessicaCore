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
 * Learning Decision
 *        ↓
 * Resolve Target Skill
 *        ↓
 * Load History
 *        ↓
 * Next Version
 *        ↓
 * Build Experience
 *        ↓
 * Atomic Save
 *        ↓
 * Approve Proposal
 *
 *
 * НЕ:
 *
 * - анализирует обучение;
 * - выбирает решение;
 * - вызывает AI;
 * - изменяет Supabase напрямую.
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
    ).trim();

}





/*
 * =========================================================
 * VERSION
 * =========================================================
 */


function getHistoryVersion(
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
                getHistoryVersion
            )
            .filter(
                Boolean
            );


    if (
        versions.length === 0
    ) {

        return 1;

    }


    return Math.max(
        ...versions
    ) + 1;

}





/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function failure(
{

    stage,

    error,

    skillId = null,

    version = null,

    proposal = null

} = {}
) {


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
            "Ошибка сохранения обучения"

    };

}





/*
 * =========================================================
 * RESOLVE TARGET
 * =========================================================
 */


function resolveLearningTarget(
{

    decision,

    proposal,

    skillId

}
) {


    const action =
        decision?.action || "";



    const proposed =
        proposal?.proposedExperience || {};




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
                    proposed.id ||
                    proposed.skillId
                )

                ||

                buildLearningSkillId(
                    proposed.name
                )


        };

    }




    /*
     * Улучшение существующего
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
                    decision.skillId
                )

        };

    }




    /*
     * Совместимость со старым Proposal
     */


    return {


        mode:
            "create",


        skillId:

            normalizeSkillId(
                skillId
            )

            ||

            normalizeSkillId(
                proposed.id ||
                proposed.skillId
            )

            ||

            buildLearningSkillId(
                proposed.name
            )

    };


}





/*
 * =========================================================
 * MAIN APPROVAL
 * =========================================================
 */


export async function approveAndSaveLearning(
{

    decision = null,

    proposal,

    skillId = "",

    confidence = 0.7

} = {}
) {



    /*
     * INPUT
     */


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {


        return failure({

            stage:
                "input",

            error:
                "Learning Proposal отсутствует"

        });

    }




    const proposedExperience =
        proposal.proposedExperience;



    if (
        !proposedExperience ||
        typeof proposedExperience !== "object"
    ) {


        return failure({

            stage:
                "input",

            proposal,

            error:
                "Нет proposedExperience"

        });

    }





    /*
     * TARGET
     */


    const target =
        resolveLearningTarget({

            decision,

            proposal,

            skillId

        });



    if (
        !target.skillId
    ) {


        return failure({

            stage:
                "skill",

            proposal,

            error:
                "Не удалось определить Skill ID"

        });

    }





    /*
     * HISTORY
     */


    let history;


    try {


        history =
            await getExperienceHistory(
                target.skillId
            );


    } catch(error) {


        return failure({

            stage:
                "history",

            proposal,

            skillId:
                target.skillId,

            error:
                error.message

        });

    }




    const version =
        getNextVersion(
            history
        );





    /*
     * BUILD
     */


    let experience;


    try {


        experience =
            buildExperienceSkill({

                proposedExperience,

                skillId:
                    target.skillId,

                version,

                confidence

            });


    } catch(error) {


        return failure({

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
     * SAVE
     */


    let saveResult;


    try {


        saveResult =
            await saveExperienceSkill(
                experience
            );


    } catch(error) {


        return failure({

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


        return failure({

            stage:
                "save",

            proposal,

            skillId:
                target.skillId,

            version,

            error:
                "Storage не подтвердил сохранение"

        });

    }





    /*
     * APPROVE ONLY AFTER SAVE
     */


    let approved;


    try {


        approved =
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





    return {

        success:true,

        stage:
            "approved",

        proposal:
            approved,

        proposalStateUpdated:true,

        experience,

        skillId:
            target.skillId,

        version,

        previousVersions:
            Array.isArray(history)
                ? history.length
                : 0,

        mode:
            target.mode,

        error:""

    };


}
