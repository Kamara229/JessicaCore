import {
    registerRootRoute
} from "./rootRoute.js";

import {
    registerHealthRoute
} from "./healthRoute.js";

import {
    registerSearchRoute
} from "./searchRoute.js";

import {
    registerFetchRoute
} from "./fetchRoute.js";

import {
    registerSolveRoute
} from "./solveRoute.js";

import {
    registerNotFoundRoute
} from "./notFoundRoute.js";


/*
 * =========================================================
 * JESSICA ROUTES
 * =========================================================
 *
 * Центральный координатор HTTP маршрутов.
 *
 *
 * server.js
 *      ↓
 * registerRoutes.js
 *      ↓
 * отдельные route-модули
 *
 *
 * ВАЖНО:
 *
 * 404 регистрируется последним.
 *
 * =========================================================
 */


export function registerRoutes(
    app
) {


    registerRootRoute(
        app
    );


    registerHealthRoute(
        app
    );


    registerSearchRoute(
        app
    );


    registerFetchRoute(
        app
    );


    registerSolveRoute(
        app
    );


    /*
     * Всегда последний.
     */


    registerNotFoundRoute(
        app
    );


}
