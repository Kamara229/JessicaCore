import {
    executeJessicaTask
} from "../../core/jessicaCore.js";

import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";
import { saveChatDebugSnapshot } from "../../storage/chatDebugSnapshot.js";


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
            let chatMirrorInput = null;
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


                chatMirrorInput = {
                    task,
                    conversation: req.body?.conversation,
                    attachments: req.body?.attachments
                };

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
                        task,
                        { conversation: req.body?.conversation, attachments: req.body?.attachments }
                    );

                // Private owner-only diagnostic; failure must never
                // prevent Jessica from replying to the Android client.
                await saveChatDebugSnapshot({...chatMirrorInput,result});


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


                // Trace the ACTUAL HTTP response, rather than only the
                // result created inside the document-analysis facade.
                res.set("Cache-Control", "no-store");
                res.on("finish", () => {
                    if (result?.mode === "document") {
                        console.log("Jessica document HTTP response:", {
                            status: res.statusCode,
                            responseId: result.responseId || null,
                            responseSha256: result.responseSha256 || null,
                            success: result.success === true
                        });
                    }
                });
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

                if (chatMirrorInput) {
                    await saveChatDebugSnapshot({
                        ...chatMirrorInput,
                        result: {success:false,text:"Внутренняя ошибка Jessica Core"}
                    });
                }


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
