import OpenAI from "openai";


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"
        })
        : null;


/*
 * =========================================================
 * JESSICA TASK PLANNER
 * =========================================================
 *
 * Planner НЕ решает задачу пользователя.
 *
 * Его задача:
 *
 * 1. понять намерение;
 * 2. определить, нужны ли инструменты;
 * 3. выбрать инструменты;
 * 4. сформировать последовательность шагов.
 *
 * Planner должен возвращать только JSON.
 */


/*
 * Инструменты, которые Jessica
 * уже умеет использовать.
 *
 * Позже список будет автоматически
 * поступать из Tool Registry.
 */
const availableTools = [

    {
        name:
            "current_time",

        description:
            "Получение точного текущего времени для города, страны или часового пояса.",

        arguments: {
            location:
                "Название города, региона или часового пояса"
        }
    },

    {
        name:
            "current_date",

        description:
            "Получение текущей даты для указанного местоположения.",

        arguments: {
            location:
                "Название города, региона или часового пояса"
        }
    },

    {
        name:
            "web_search",

        description:
            "Поиск актуальной информации в интернете.",

        arguments: {
            query:
                "Поисковый запрос"
        }
    },

    {
        name:
            "web_fetch",

        description:
            "Загрузка содержимого конкретной веб-страницы.",

        arguments: {
            url:
                "URL страницы"
        }
    }

];


function buildToolsDescription() {

    return availableTools
        .map(
            tool => {

                return JSON.stringify(
                    tool,
                    null,
                    2
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * Удаляем markdown-блоки,
 * если модель всё же вернула ```json.
 */
function cleanJsonText(
    text
) {

    return text
        .replace(
            /```json/gi,
            ""
        )
        .replace(
            /```/g,
            ""
        )
        .trim();

}


/*
 * Проверяем минимальную структуру плана.
 */
function validatePlan(
    plan
) {

    if (
        !plan ||
        typeof plan !== "object"
    ) {

        return false;

    }


    if (
        typeof plan.intent !==
        "string"
    ) {

        return false;

    }


    if (
        typeof plan.requiresTools !==
        "boolean"
    ) {

        return false;

    }


    if (
        !Array.isArray(
            plan.steps
        )
    ) {

        return false;

    }


    return true;

}


/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(
    task
) {

    if (!groq) {

        return {
            success: false,
            text:
                "Groq Planner не настроен"
        };

    }


    try {

        const toolsDescription =
            buildToolsDescription();


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Planner системы Jessica Core. " +

                        "Ты НЕ отвечаешь пользователю и НЕ решаешь задачу напрямую. " +

                        "Ты анализируешь задачу и создаёшь план выполнения. " +

                        "Jessica имеет набор инструментов. " +

                        "Используй инструмент только тогда, когда он действительно нужен. " +

                        "Если задачу можно решить знаниями AI без актуальных внешних данных, " +
                        "requiresTools должен быть false, а steps должен быть пустым массивом. " +

                        "Если нужны актуальные или внешние данные, выбери подходящий инструмент. " +

                        "Не пытайся вручную угадывать ответ вместо инструмента. " +

                        "Если пользователь спрашивает текущее время, используй current_time. " +

                        "Если пользователь спрашивает текущую дату, используй current_date. " +

                        "Если нужны свежие данные из интернета, используй web_search. " +

                        "Если нужно прочитать конкретную страницу, используй web_fetch. " +

                        "Возвращай ТОЛЬКО валидный JSON. " +

                        "Не используй markdown. " +

                        "Формат ответа: " +

                        JSON.stringify({

                            intent:
                                "краткое название намерения",

                            requiresTools:
                                true,

                            reasoningSummary:
                                "очень краткое объяснение выбранного маршрута",

                            steps: [

                                {
                                    tool:
                                        "название инструмента",

                                    arguments: {}
                                }

                            ]

                        })
                    ),

                input:
                    (
                        `ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n` +
                        `${toolsDescription}\n\n` +

                        `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                        `${task}`
                    ),

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const raw =
            response.output_text
                ?.trim();


        if (!raw) {

            return {
                success: false,
                text:
                    "Planner вернул пустой ответ"
            };

        }


        let plan;


        try {

            plan =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );

        } catch {

            console.error(
                "Planner invalid JSON:",
                raw
            );


            return {
                success: false,
                text:
                    "Planner вернул некорректный JSON"
            };

        }


        if (
            !validatePlan(
                plan
            )
        ) {

            console.error(
                "Planner invalid structure:",
                plan
            );


            return {
                success: false,
                text:
                    "Planner вернул некорректную структуру плана"
            };

        }


        return {
            success: true,
            plan
        };


    } catch (error) {

        console.error(
            "Planner error:",
            error
        );


        return {
            success: false,

            status:
                error?.status || 0,

            text:
                "Planner не смог проанализировать задачу"
        };

    }

}
