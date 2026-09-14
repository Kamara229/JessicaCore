/*
 * =========================================================
 * ROOT ROUTE
 * =========================================================
 *
 * Корневой endpoint backend Jessica.
 *
 * Используется для простой проверки,
 * что сервер запущен и отвечает.
 *
 * =========================================================
 */


export function registerRootRoute(
    app
) {


    app.get(
        "/",

        (
            req,
            res
        ) => {


            return res.json({

                success:
                    true,

                service:
                    "Jessica Core",

                version:
                    "1.0",

                status:
                    "running"

            });


        }
    );


}
