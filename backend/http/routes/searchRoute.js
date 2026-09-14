import {
    executeTool
} from "../../tools/toolRegistry.js";

import {
    requireJessicaAuthorization
} from "../auth/jessicaAuth.js";


/*
 * =========================================================
 * DIRECT SEARCH ROUTE
 * =========================================================
 */


export function registerSearchRoute(
    app
) {

    app.post(
        "/api/search",

        requireJessicaAuthorization,

        async (
            req,
            res
        ) => {

            try {


                const query =
                    typeof req.body?.query === "string"
                        ? req.body.query.trim()
                        : "";


                if (!query) {

                    return res
                        .status(
                            400
                        )
                        .json({

                            success:
                                false,

                            text:
                                "Не указан поисковый запрос"

                        });

                }


                const result =
                    await executeTool(

                        "web_search",

                        {
                            query
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
                    "/api/search error:",
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
                            "Внутренняя ошибка интернет-поиска"

                    });

            }

        }
    );

}
