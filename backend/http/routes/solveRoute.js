import {
    executeJessicaTask
} from "../../core/jessicaCore.js";

import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";
import { saveChatDebugSnapshot } from "../../storage/chatDebugSnapshot.js";
import { localizeResponse, normalizeOutputLanguage, requestedTranslation } from "../../core/language/responseLanguageService.js";
import {withResponseLanguage} from "../../core/language/responseLanguageContext.js";
import {enforcePresentationLanguage} from "../../core/language/strictPresentationLanguage.js";
import {isBareSourceOnly} from "../../core/language/codeAwareLocale.js";


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


                const outputLanguage = normalizeOutputLanguage(req.body?.outputLanguage);
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
                                (outputLanguage === "en" ? "Task not provided" : "Задача не указана")

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
                                (outputLanguage === "en" ? "Task is too long" : "Задача слишком большого объёма")

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


                const coreResult =
                    await withResponseLanguage(outputLanguage, () => executeJessicaTask(
                        task,
                        { conversation: req.body?.conversation, attachments: req.body?.attachments, outputLanguage }
                    ));

                const preserveBareCode = coreResult?.mode === "code" &&
                    isBareSourceOnly(coreResult?.text);
                const localized = coreResult?.mode === "translation"
                    ? {...coreResult, outputLanguage, localizationStatus:"explicit-translation"}
                    : coreResult?.mode === "conversation_ack"
                        ? {...coreResult, outputLanguage, localizationStatus:"deterministic-chat-ack"}
                    : preserveBareCode
                        ? {...coreResult, outputLanguage, localizationStatus:"bare-code-preserved"}
                        : await localizeResponse(coreResult, task, outputLanguage);
                const result = enforcePresentationLanguage(localized, {
                    task,
                    language:outputLanguage,
                    override:requestedTranslation(task)
                });
                console.log("Jessica response locale:", {
                    requested: outputLanguage,
                    mode: result?.mode || null,
                    status: result?.localizationStatus || "none",
                    textLength: String(result?.text || "").length
                });

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
                            (normalizeOutputLanguage(req.body?.outputLanguage) === "en"
                                ? "Jessica Core internal error" : "Внутренняя ошибка Jessica Core")

                    });

            }

        }
    );

}
