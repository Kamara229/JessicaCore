/*
 * =========================================================
 * JESSICA LEARNING END-TO-END TEST
 * =========================================================
 *
 * Проверяет реальную цепочку:
 *
 * HTTP
 * ↓
 * /api/learning/test
 * ↓
 * Learning Core
 * ↓
 * Analyzer
 * ↓
 * Parser
 * ↓
 * Validator
 * ↓
 * Learning Proposal
 *
 *
 * ВАЖНО:
 *
 * Этот тест НЕ подтверждает Proposal
 * и НЕ сохраняет Experience.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const BASE_URL =
    String(
        process.env.JESSICA_BASE_URL ||
        "https://jessicacore.onrender.com"
    )
        .trim()
        .replace(
            /\/+$/,
            ""
        );


const token =
    String(
        process.env.JESSICA_APP_TOKEN || ""
    ).trim();


if (!token) {

    throw new Error(
        "JESSICA_APP_TOKEN не настроен"
    );

}


/*
 * =========================================================
 * TEST DATA
 * =========================================================
 */


const payload = {

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
 * REQUEST
 * =========================================================
 */


async function run() {


    console.log(
        "Jessica Learning test started..."
    );


    const response =
        await fetch(
            `${BASE_URL}/api/learning/test`,
            {

                method:
                    "POST",

                headers: {

                    "Content-Type":
                        "application/json",

                    "X-Jessica-Token":
                        token

                },

                body:
                    JSON.stringify(
                        payload
                    )

            }
        );


    const text =
        await response.text();


    let result;


    try {

        result =
            JSON.parse(
                text
            );

    } catch {

        result = {
            raw:
                text
        };

    }


    console.log(
        "HTTP status:",
        response.status
    );


    console.log(
        JSON.stringify(
            result,
            null,
            2
        )
    );


    if (!response.ok) {

        process.exitCode =
            1;

        return;

    }


    if (
        result?.success !== true
    ) {

        process.exitCode =
            1;

        return;

    }


    console.log(
        "Jessica Learning test completed."
    );

}


/*
 * =========================================================
 * START
 * =========================================================
 */


run()
    .catch(
        error => {

            console.error(
                "Jessica Learning test failed:",
                error
            );

            process.exitCode =
                1;

        }
    );
