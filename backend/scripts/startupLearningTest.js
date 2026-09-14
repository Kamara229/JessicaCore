import {
    createLearning
} from "../experience/learning/learningCore.js";


/*
 * =========================================================
 * JESSICA STARTUP LEARNING TEST
 * =========================================================
 *
 * Временный диагностический тест Learning Core.
 *
 * Запускается только если:
 *
 * RUN_LEARNING_TEST_ON_START=true
 *
 *
 * Не использует HTTP.
 * Не требует JESSICA_APP_TOKEN.
 * Не сохраняет Experience.
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
 * CHECK FLAG
 * =========================================================
 */


function isTestEnabled() {

    return (
        String(
            process.env.RUN_LEARNING_TEST_ON_START || ""
        )
            .trim()
            .toLowerCase() ===
        "true"
    );

}


/*
 * =========================================================
 * RUN TEST
 * =========================================================
 */


export async function runStartupLearningTest() {


    if (
        !isTestEnabled()
    ) {

        return;

    }


    console.log(
        "========================================"
    );

    console.log(
        "Jessica Startup Learning Test: START"
    );

    console.log(
        "========================================"
    );


    try {


        const result =
            await createLearning(
                TEST_DATA
            );


        console.log(
            "Jessica Startup Learning Test result:"
        );


        console.log(
            JSON.stringify(
                result,
                null,
                2
            )
        );


        console.log(
            "========================================"
        );

        console.log(
            "Jessica Startup Learning Test: END"
        );

        console.log(
            "========================================"
        );


    } catch (error) {


        console.error(
            "Jessica Startup Learning Test failed:",
            error
        );


    }

}
