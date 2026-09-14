import {
    executeJessicaTask
} from "../../core/jessicaCore.js";

import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";


/*
 * =========================================================
 * JESSICA SOLVE ROUTE
 * =========================================================
 *
 * Главная пользовательская точка входа:
 *
 * Android
 *      ↓
 * /api/solve
 *      ↓
 * Jessica Core
 *
 * =========================================================
 */


const MAX_TASK_LENGTH =
    20000;


/*
 * =========================================================
 * REGISTER ROUTE
 * =========================================================
 */


export function registerSolveRoute(
    app
) {

    app.post(
        "/api/solve",

        requireJessicaAuthorization,

        async (
            req,
            res
        ) => {

            try {


                /*
                 * =================================================
                 * TASK
                 * =================================================
                 */


                const task =
                    typeof req.body?.task === "string"
                        ? req.body.task.trim()
                        : "";


                if (!task) {

                    return res
                        .status(
                            400
                        )
                        .json({

                            success:
                                false,

                            text:
                                "Задача не указана"

                        });

                }


                /*
                 * =================================================
                 * SIZE LIMIT
                 * =================================================
                 */


                if (
                    task.length >
                    MAX_TASK_LENGTH
                ) {

                    return res
                        .status(
                            413
                        )
                        .json({

                            success:
                                false,

                            text:
                                "Задача слишком большого объёма"

                        });

                }


                console.log(
                    "Jessica task:",
                    task
                );


                /*
                 * =================================================
                 * JESSICA CORE
                 * =================================================
                 */


                const result =
                    await executeJessicaTask(
                        task
                    );


                /*
                 * =================================================
                 * RESPONSE
                 * =================================================
                 *
                 * Пока любой корректно обработанный результат
                 * Jessica возвращается HTTP 200.
                 *
                 * Android ориентируется на:
                 *
                 * success
                 * needsClarification
                 *
                 * =================================================
                 */


                return res
                    .status(
                        200
                    )
                    .json(
                        result
                    );


            } catch (error) {


                console.error(
                    "/api/solve error:",
                    error
                );


                return res
                    .status(
                        500
                    )
                    .json({

                        success:
                            false,

                        stage:
                            "server",

                        text:
                            "Внутренняя ошибка Jessica Core"

                    });

            }

        }
    );

}
