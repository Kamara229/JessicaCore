/*
 * =========================================================
 * JESSICA HTTP AUTH
 * =========================================================
 *
 * Авторизация запросов Android → Backend.
 *
 * Используется защищёнными endpoint:
 *
 * /api/solve
 * /api/search
 * /api/fetch
 *
 * =========================================================
 */


/*
 * =========================================================
 * TOKEN
 * =========================================================
 */


function getJessicaToken() {

    return String(
        process.env.JESSICA_APP_TOKEN || ""
    ).trim();

}


/*
 * =========================================================
 * CHECK AUTHORIZATION
 * =========================================================
 */


export function checkJessicaAuthorization(
    req
) {

    const jessicaToken =
        getJessicaToken();


    if (!jessicaToken) {

        return {

            success:
                false,

            status:
                503,

            text:
                "Авторизация Jessica не настроена на сервере"

        };

    }


    const appToken =
        String(
            req.get(
                "X-Jessica-Token"
            ) || ""
        ).trim();


    if (
        appToken !==
        jessicaToken
    ) {

        return {

            success:
                false,

            status:
                401,

            text:
                "Неавторизованный запрос"

        };

    }


    return {

        success:
            true

    };

}


/*
 * =========================================================
 * AUTH MIDDLEWARE
 * =========================================================
 */


export function requireJessicaAuthorization(
    req,
    res,
    next
) {

    const auth =
        checkJessicaAuthorization(
            req
        );


    if (!auth.success) {

        return res
            .status(
                auth.status
            )
            .json({

                success:
                    false,

                text:
                    auth.text

            });

    }


    next();

}
