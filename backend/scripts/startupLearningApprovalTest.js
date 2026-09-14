import {
    createLearning,
    approveLearning
} from "../experience/learning/learningCore.js";


/*
 * =========================================================
 * JESSICA STARTUP LEARNING APPROVAL TEST
 * =========================================================
 *
 * Проверяет полный цикл:
 *
 * correction
 *      ↓
 * createLearning()
 *      ↓
 * Validators
 *      ↓
 * PENDING_APPROVAL
 *      ↓
 * approveLearning()
 *      ↓
 * Experience Skill
 *      ↓
 * Supabase
 *
 *
 * ВАЖНО:
 *
 * Скрипт запускается ТОЛЬКО если:
 *
 * RUN_LEARNING_APPROVAL_TEST_ON_START=true
 *
 *
 * Обычный:
 *
 * RUN_LEARNING_TEST_ON_START
 *
 * к этому тесту отношения не имеет.
 *
 * =========================================================
 */


/*
 * =========================================================
 * TEST DATA
 * =========================================================
 */


const TEST_DATA = {

    task:
        "Найди официальный сайт Blender и укажи ссылку",

    previousAnswer:
        "Официальный сайт Blender можно найти через один из результатов поиска.",

    correction:
        [
            "Для таких задач сначала найди официальный сайт через поиск.",
            "Затем обязательно проверь, что домен действительно принадлежит нужной организации или продукту.",
            "Не используй Википедию, каталоги и сторонние сайты как официальный источник, если доступен сайт самой организации.",
            "Для Blender правильный официальный домен — blender.org."
        ].join(
            " "
        ),

    correctedAnswer:
        "Официальный сайт Blender: https://www.blender.org/"

};


/*
 * =========================================================
 * FLAG
 * =========================================================
 */


function isTestEnabled() {

    return (
        String(
            process.env
                .RUN_LEARNING_APPROVAL_TEST_ON_START ||
            ""
        )
            .trim()
            .toLowerCase() ===
        "true"
    );

}


/*
 * =========================================================
 * RUN
 * =========================================================
 */


export async function runStartupLearningApprovalTest() {


    if (
        !isTestEnabled()
    ) {

        return;

    }


    console.log(
        "========================================"
    );

    console.log(
        "Jessica Learning Approval Test: START"
    );

    console.log(
        "========================================"
    );


    try {


        /*
         * =================================================
         * 1. CREATE LEARNING
         * =================================================
         */


        const learningResult =
            await createLearning(
                TEST_DATA
            );


        console.log(
            "Jessica Learning Approval proposal:"
        );


        console.log(
            JSON.stringify(
                {
                    success:
                        learningResult?.success,

                    stage:
                        learningResult?.stage,

                    reusable:
                        learningResult?.reusable,

                    needsClarification:
                        learningResult
                            ?.needsClarification,

                    readyForApproval:
                        learningResult
                            ?.readyForApproval,

                    proposalId:
                        learningResult
                            ?.proposal
                            ?.id || null,

                    understanding:
                        learningResult
                            ?.proposal
                            ?.understanding || ""
                },
                null,
                2
            )
        );


        /*
         * =================================================
         * 2. SAFETY CHECK
         * =================================================
         */


        if (
            learningResult?.success !== true ||
            learningResult?.readyForApproval !== true
        ) {


            console.log(
                "Jessica Learning Approval Test: proposal rejected by safety checks"
            );


            return;

        }


        /*
         * =================================================
         * 3. APPROVE
         * =================================================
         *
         * В этот блок выполнение попадёт
         * только при отдельном approval-флаге.
         *
         * =================================================
         */


        const approvalResult =
            await approveLearning({

                learningResult,

                confidence:
                    0.7

            });


        /*
         * =================================================
         * 4. RESULT
         * =================================================
         */


        console.log(
            "Jessica Learning Approval result:"
        );


        console.log(
            JSON.stringify(
                approvalResult,
                null,
                2
            )
        );


    } catch (error) {


        console.error(
            "Jessica Learning Approval Test failed:",
            error
        );


    } finally {


        console.log(
            "========================================"
        );

        console.log(
            "Jessica Learning Approval Test: END"
        );

        console.log(
            "========================================"
        );


    }

}
