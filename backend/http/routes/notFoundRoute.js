/*
 * =========================================================
 * 404 ROUTE
 * =========================================================
 */


export function registerNotFoundRoute(
    app
) {

    app.use(
        (
            req,
            res
        ) => {

            return res
                .status(
                    404
                )
                .json({

                    success:
                        false,

                    text:
                        "Endpoint не найден"

                });

        }
    );

}
