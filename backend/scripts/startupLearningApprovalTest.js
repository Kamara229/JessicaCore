import {
    createLearning,
    approveLearning
} from "../experience/learning/learningCore.js";

import {
    getExperienceHistory
} from "../experience/storage/experienceStorage.js";


/*
 * =========================================================
 * JESSICA STARTUP LEARNING APPROVAL TEST
 * =========================================================
 *
 * Одноразовый тест полного цикла обучения:
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
 * Experience Skill v1
 *      ↓
 * Supabase
 *
 *
 * ВАЖНО:
 *
 * Этот тест предназначен ТОЛЬКО
 * для первого сохранения тестового Skill.
 *
 *
 * Перед обучением обязательно проверяется:
 *
 * TEST_SKILL_ID
 *      ↓
 * Experience History
 *      ↓
 * история пустая?
 *
 * ДА:
 *      разрешаем тест
 *
 * НЕТ:
 *      STOP
 *
 *
 * Поэтому повторный startup
 * не должен создавать:
 *
 * v2
 * v3
 * v4
 * ...
 *
 *
 * Скрипт запускается только если:
 *
 * RUN_LEARNING_APPROVAL_TEST_ON_START=true
 *
 * =========================================================
 */


/*
 * =========================================================
 * TEST SKILL
 * =========================================================
 *
 * Используем фиксированный ID.
 *
 * Это принципиально важно:
 *
 * мы не позволяем AI каждый раз
 * генерировать новый ID и обходить
 * защиту от повторного теста.
 *
 * =========================================================
 */


const TEST_SKILL_ID =
    "official-website-verification";


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
 * CHECK EXISTING HISTORY
 * =========================================================
 *
 * Возвращает:
 *
 * {
 *   exists,
 *   history
 * }
 *
 * =========================================================
 */


async function checkExistingSkill() {


    const history =
        await getExperienceHistory(
            TEST_SKILL_ID
        );


    const safeHistory =
        Array.isArray(
            history
        )
            ? history
            : [];


    return {

        exists:
            safeHistory.length > 0,

        history:
            safeHistory

    };

}


/*
 * =========================================================
 * RUN
 * =========================================================
 */


export async function runStartupLearningApprovalTest() {


    /*
     * =====================================================
     * FLAG CHECK
     * =====================================================
     */


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


    console.log(
        "Jessica Learning Approval Test skill:",
        TEST_SKILL_ID
    );


    try {


        /*
         * =================================================
         * 1. REPEAT PROTECTION
         * =================================================
         *
         * Сначала проверяем History.
         *
         * Если хотя бы одна версия
         * этого тестового Skill уже существует,
         * повторное обучение запрещаем.
         *
         * =================================================
         */


        const existingSkill =
            await checkExistingSkill();


        if (
            existingSkill.exists
        ) {


            console.log(
                "Jessica Learning Approval Test: SKIPPED"
            );


            console.log(
                `Skill "${TEST_SKILL_ID}" уже существует`
            );


            console.log(
                "Existing versions:",
                existingSkill.history.length
            );


            console.log(
                "Повторное сохранение тестового Skill запрещено"
            );


            return;

        }


        console.log(
            "Jessica Learning Approval Test: existing Skill not found"
        );


        /*
         * =================================================
         * 2. CREATE LEARNING
         * =================================================
         */


        const learningResult =
            await createLearning(
                TEST_DATA
            );


        /*
         * =================================================
         * 3. PROPOSAL LOG
         * =================================================
         */


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
                            ?.understanding || "",

                    proposedExperience:
                        learningResult
                            ?.proposal
                            ?.proposedExperience ||
                        null

                },
                null,
                2
            )
        );


        /*
         * =================================================
         * 4. SAFETY CHECK
         * =================================================
         *
         * До реального сохранения допускается
         * только Learning Result, прошедший:
         *
         * - Parser;
         * - Structure Validator;
         * - Grounding Validator;
         * - Generalization Validator;
         * - reusable check;
         * - clarification check.
         *
         * =================================================
         */


        if (
            learningResult?.success !== true ||
            learningResult?.readyForApproval !== true
        ) {


            console.log(
                "Jessica Learning Approval Test: proposal rejected by safety checks"
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

                        error:
                            learningResult?.error || ""

                    },
                    null,
                    2
                )
            );


            return;

        }


        /*
         * =================================================
         * 5. SECOND REPEAT CHECK
         * =================================================
         *
         * Между первым чтением History
         * и окончанием AI-анализа
         * прошло некоторое время.
         *
         * Поэтому непосредственно перед
         * записью проверяем History ещё раз.
         *
         * Это дополнительная защита
         * от параллельного запуска.
         *
         * =================================================
         */


        const beforeApproval =
            await checkExistingSkill();


        if (
            beforeApproval.exists
        ) {


            console.log(
                "Jessica Learning Approval Test: STOP before save"
            );


            console.log(
                `Skill "${TEST_SKILL_ID}" появился во время выполнения теста`
            );


            console.log(
                "Сохранение отменено"
            );


            return;

        }


        /*
         * =================================================
         * 6. APPROVE
         * =================================================
         *
         * Передаём ФИКСИРОВАННЫЙ skillId.
         *
         * Это не позволяет Analyzer
         * создать новый ID при каждом запуске.
         *
         * =================================================
         */


        const approvalResult =
            await approveLearning({

                learningResult,

                skillId:
                    TEST_SKILL_ID,

                confidence:
                    0.7

            });


        /*
         * =================================================
         * 7. RESULT
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


        /*
         * =================================================
         * 8. FINAL CHECK
         * =================================================
         *
         * После успешной записи проверяем,
         * появилась ли версия в History.
         *
         * =================================================
         */


        if (
            approvalResult?.success === true
        ) {


            const finalHistory =
                await getExperienceHistory(
                    TEST_SKILL_ID
                );


            console.log(
                "Jessica Learning Approval final history:"
            );


            console.log(
                JSON.stringify(
                    {

                        skillId:
                            TEST_SKILL_ID,

                        versions:
                            Array.isArray(
                                finalHistory
                            )
                                ? finalHistory.length
                                : 0,

                        savedVersion:
                            approvalResult?.version ||
                            null

                    },
                    null,
                    2
                )
            );


        }


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
