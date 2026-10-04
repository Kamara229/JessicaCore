/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST STORAGE
 * =========================================================
 */


import {
    createLearningQueueItem
} from "../../core/learning/learningQueue.js";


import {
    saveLearningQueueItem
} from "../../core/learning/learningQueueStorage.js";


import {
    findLearningProposalByQueueItemId
} from "../../core/learning/learningProposalStorage.js";


import {
    getExperienceHistory,
    disableExperienceSkill
} from "../../experience/storage/experienceStorage.js";


import {
    loadExperienceSkill
} from "../../experience/storage/supabaseExperienceStore.js";


import {
    AUTONOMOUS_TEST_SKILL_ID
} from "./autonomousTestConstants.js";





export async function ensureAutonomousTestCanRun()
{

    const history =

        await getExperienceHistory(
            AUTONOMOUS_TEST_SKILL_ID
        );


    const versions =

        Array.isArray(history)

            ? history

            : [];


    return {

        canRun:

            versions.length === 0,

        history:
            versions

    };

}





export async function queueAutonomousTestEvent(
    event
) {

    const queueItem =

        createLearningQueueItem(
            event
        );


    if(
        !queueItem
    ){

        return {

            success:
                false,

            queueItem:
                null,

            error:
                "Queue Item не создан"

        };

    }


    const saved =

        await saveLearningQueueItem(
            queueItem
        );


    if(
        !saved?.success
    ){

        return {

            success:
                false,

            queueItem,

            error:

                saved?.error

                ||

                "Queue Item не сохранён"

        };

    }


    return {

        success:
            true,

        queueItem:

            saved.item

            ||

            queueItem

    };

}





export async function inspectAutonomousTestState(
    queueItemId
) {

    const proposalResult =

        await findLearningProposalByQueueItemId(
            queueItemId
        );


    const history =

        await getExperienceHistory(
            AUTONOMOUS_TEST_SKILL_ID
        );


    const activeSkill =

        await loadExperienceSkill(
            AUTONOMOUS_TEST_SKILL_ID
        );


    return {

        proposal:

            proposalResult?.proposal

            ||

            null,

        proposalLookup:

            proposalResult,

        history:

            Array.isArray(history)

                ? history

                : [],

        activeSkill:

            activeSkill

            ||

            null

    };

}





export async function cleanupAutonomousTestSkill()
{

    try {


        const result =

            await disableExperienceSkill(
                AUTONOMOUS_TEST_SKILL_ID
            );


        return {

            success:

                result?.success === true,

            result

        };


    }catch(error){


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Diagnostic Skill cleanup failed"

        };

    }

}
