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


const openai =
    new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
    });


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


app.get(
    "/api/health",
    (req, res) => {

        res.json({
            success: true,
            status: "ok"
        });

    }
);


app.post(
    "/api/solve",
    async (req, res) => {

        try {

            const task =
                req.body?.task?.trim();


            if (!task) {

                return res
                    .status(400)
                    .json({
                        success: false,
                        text: "Задача не указана"
                    });

            }


            if (!process.env.OPENAI_API_KEY) {

                return res
                    .status(500)
                    .json({
                        success: false,
                        text: "OPENAI_API_KEY не настроен на сервере"
                    });

            }


            const response =
                await openai.responses.create({

                    model:
                        "gpt-5.6",

                    input: [
                        {
                            role: "system",
                            content:
                                "Ты являешься AI-ядром системы Jessica Core. " +
                                "Выполняй задачу пользователя точно и полезно. " +
                                "Отвечай на языке задачи."
                        },
                        {
                            role: "user",
                            content:
                                task
                        }
                    ]

                });


            const answer =
                response.output_text?.trim();


            if (!answer) {

                return res
                    .status(502)
                    .json({
                        success: false,
                        text: "Модель вернула пустой ответ"
                    });

            }


            return res.json({
                success: true,
                text: answer
            });

        } catch (error) {

            console.error(
                "Jessica AI error:",
                error
            );


            return res
                .status(500)
                .json({
                    success: false,
                    text:
                        error?.message
                            ? `Ошибка AI: ${error.message}`
                            : "Неизвестная ошибка AI"
                });

        }

    }
);


const port =
    process.env.PORT || 3000;


app.listen(
    port,
    () => {

        console.log(
            `Jessica Backend запущен на порту ${port}`
        );

    }
);
