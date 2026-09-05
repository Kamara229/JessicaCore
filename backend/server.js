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
 * Создаём клиент OpenAI только если
 * ключ действительно настроен.
 *
 * Благодаря этому backend сможет
 * запуститься и показать /api/health
 * даже до добавления API-ключа.
 */
const openai =
    process.env.OPENAI_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.OPENAI_API_KEY
        })
        : null;


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
 * Проверка состояния сервера.
 */
app.get(
    "/api/health",
    (req, res) => {

        res.json({
            success: true,
            service: "Jessica Backend",
            status: "ok",
            aiConfigured:
                openai !== null
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
                        text: "Задача не указана"
                    });

            }


            /*
             * Проверяем наличие API-ключа.
             */
            if (!openai) {

                return res
                    .status(503)
                    .json({
                        success: false,
                        text:
                            "AI Engine не настроен: отсутствует OPENAI_API_KEY"
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
             * Возвращаем результат Android-приложению.
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


            const errorMessage =
                error instanceof Error
                    ? error.message
                    : "Неизвестная ошибка";


            return res
                .status(500)
                .json({
                    success: false,
                    text:
                        `Ошибка AI: ${errorMessage}`
                });

        }

    }
);


/*
 * Render автоматически передаёт PORT.
 * Локально используется порт 3000.
 */
const port =
    Number(
        process.env.PORT
    ) || 3000;


/*
 * 0.0.0.0 нужен для нормальной
 * работы Web Service на Render.
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
