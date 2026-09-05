import express from "express";
import OpenAI from "openai";
import dotenv from "dotenv";


dotenv.config();


const app =
    express();


app.use(
    express.json({
        limit: "1mb"
    })
);


/*
 * Клиент OpenAI создаётся только
 * если API-ключ настроен.
 */
const openai =
    process.env.OPENAI_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.OPENAI_API_KEY
        })
        : null;


/*
 * Секретный токен Jessica.
 *
 * Он хранится только в переменных
 * окружения Render.
 */
const jessicaToken =
    process.env.JESSICA_APP_TOKEN || "";


/*
 * Главная страница backend.
 */
app.get(
    "/",
    (req, res) => {

        res.json({
            success: true,
            service: "Jessica Backend",
            status: "online"
        });

    }
);


/*
 * Проверка состояния backend.
 *
 * Здесь специально НЕ показываем
 * само значение секретных ключей.
 */
app.get(
    "/api/health",
    (req, res) => {

        res.json({
            success: true,
            service: "Jessica Backend",
            status: "ok",
            aiConfigured:
                openai !== null,
            appAuthConfigured:
                jessicaToken.length > 0
        });

    }
);


/*
 * Выполнение задачи Jessica.
 */
app.post(
    "/api/solve",
    async (req, res) => {

        try {

            /*
             * Проверяем, что секретный
             * токен вообще настроен
             * на сервере.
             */
            if (!jessicaToken) {

                return res
                    .status(503)
                    .json({
                        success: false,
                        text:
                            "Авторизация Jessica не настроена на сервере"
                    });

            }


            /*
             * Получаем токен,
             * присланный Android-приложением.
             */
            const appToken =
                req.get(
                    "X-Jessica-Token"
                ) || "";


            /*
             * Не разрешаем доступ
             * без правильного токена.
             */
            if (
                appToken !==
                jessicaToken
            ) {

                return res
                    .status(401)
                    .json({
                        success: false,
                        text:
                            "Неавторизованный запрос"
                    });

            }


            const task =
                typeof req.body?.task === "string"
                    ? req.body.task.trim()
                    : "";


            /*
             * Проверяем текст задачи.
             */
            if (!task) {

                return res
                    .status(400)
                    .json({
                        success: false,
                        text:
                            "Задача не указана"
                    });

            }


            /*
             * Проверяем наличие
             * OpenAI API-ключа.
             */
            if (!openai) {

                return res
                    .status(503)
                    .json({
                        success: false,
                        text:
                            "AI Engine не настроен"
                    });

            }


            /*
             * Отправляем задачу модели.
             */
            const response =
                await openai.responses.create({

                    model:
                        "gpt-5.6-sol",

                    instructions:
                        "Ты являешься AI-ядром системы Jessica Core. " +
                        "Выполняй задачу пользователя точно, полезно и по существу. " +
                        "Отвечай на языке, на котором сформулирована задача. " +
                        "Не утверждай, что выполнил действия во внешних системах, " +
                        "если фактически у тебя нет соответствующего инструмента.",

                    input:
                        task

                });


            /*
             * Берём итоговый текст ответа.
             */
            const answer =
                response.output_text
                    ?.trim();


            if (!answer) {

                return res
                    .status(502)
                    .json({
                        success: false,
                        text:
                            "Модель вернула пустой ответ"
                    });

            }


            /*
             * Возвращаем ответ Jessica.
             */
            return res.json({
                success: true,
                text: answer
            });

        } catch (error) {

            console.error(
                "Jessica AI error:",
                error
            );


            /*
             * Наружу не отдаём
             * подробности внутренней ошибки.
             */
            return res
                .status(500)
                .json({
                    success: false,
                    text:
                        "Внутренняя ошибка AI-сервера"
                });

        }

    }
);


/*
 * Render автоматически передаёт PORT.
 */
const port =
    Number(
        process.env.PORT
    ) || 3000;


/*
 * 0.0.0.0 нужен для Render.
 */
app.listen(
    port,
    "0.0.0.0",
    () => {

        console.log(
            `Jessica Backend запущен на порту ${port}`
        );

    }
);
