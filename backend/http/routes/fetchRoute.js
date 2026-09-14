import {
    executeTool
} from "../../tools/toolRegistry.js";

import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";


/*
 * =========================================================
 * DIRECT FETCH ROUTE
 * =========================================================
 */


export function registerFetchRoute(
    app
) {

    app.post(
        "/api/fetch",

        requireJessicaAuthorization,

        async (
            req,
            res
        ) => {

            try {


                const url =
                    typeof req.body?.url === "string"
                        ? req.body.url.trim()
                        : "";


                if (!url) {

                    return res
                        .status(
                            400
                        )
                        .json({

                            success:
                                false,

                            text:
                                "Не указан URL"

                        });

                }


                const result =
                    await executeTool(

                        "web_fetch",

                        {
                            url
                        }

                    );


                return res
                    .status(
                        result.success
                            ? 200
                            : 502
                    )
                    .json(
                        result
                    );


            } catch (error) {


                console.error(
                    "/api/fetch error:",
                    error
                );


                return res
                    .status(
                        500
                    )
                    .json({

                        success:
                            false,

                        text:
                            "Внутренняя ошибка загрузки страницы"

                    });

            }

        }
    );

}
