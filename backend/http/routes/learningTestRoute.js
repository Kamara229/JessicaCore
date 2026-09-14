import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";

import {
    createLearning
} from "../../experience/learning/learningCore.js";


/*
 * =========================================================
 * JESSICA LEARNING TEST ROUTE
 * =========================================================
 *
 * Временный защищённый маршрут
 * для проверки Learning Core.
 *
 *
 * Endpoint:
 *
 * POST /api/learning/test
 *
 *
 * Он умеет только:
 *
 * correction
 *      ↓
 * Learning Analyzer
 *      ↓
 * Learning Proposal
 *      ↓
 * PENDING_APPROVAL
 *
 *
 * ВАЖНО:
 *
 * Этот маршрут НЕ:
 *
 * - подтверждает обучение;
 * - сохраняет Skill;
 * - изменяет Supabase Experience;
 * - создаёт новую версию;
 * - повторно выполняет задачу.
 *
 * =========================================================
 */


/*
 * =========================================================
 * REGISTER ROUTE
 * =========================================================
 */


export function registerLearningTestRoute(
    app
) {


    app.post(

        "/api/learning/test",

        requireJessicaAuthorization,

        async (
            req,
            res
        ) => {


            /*
             * =============================================
             * INPUT
             * =============================================
             */


            const task =
                String(
                    req.body?.task || ""
                ).trim();


            const previousAnswer =
                String(
                    req.body?.previousAnswer || ""
                ).trim();


            const correction =
                String(
                    req.body?.correction || ""
                ).trim();


            const correctedAnswer =
                String(
                    req.body?.correctedAnswer || ""
                ).trim();


            /*
             * =============================================
             * VALIDATION
             * =============================================
             */


            if (!task) {

                return res
                    .status(
                        400
                    )
                    .json({

                        success:
                            false,

                        stage:
                            "input",

                        error:
                            "Поле task обязательно"

                    });

            }


            if (!correction) {

                return res
                    .status(
                        400
                    )
                    .json({

                        success:
                            false,

                        stage:
                            "input",

                        error:
                            "Поле correction обязательно"

                    });

            }


            /*
             * =============================================
             * LEARNING
             * =============================================
             */


            try {


                console.log(
                    "Jessica Learning test:",
                    task
                );


                const result =
                    await createLearning({

                        task,

                        previousAnswer,

                        correction,

                        correctedAnswer

                    });


                /*
                 * Learning Result может быть неуспешным
                 * с точки зрения Analyzer,
                 * но сам HTTP-запрос при этом выполнен.
                 *
                 * Поэтому возвращаем 200,
                 * чтобы видеть полный результат анализа.
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
                    "Jessica Learning test route error:",
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

                        error:
                            "Непредвиденная ошибка Learning Test"

                    });

            }


        }

    );


}
